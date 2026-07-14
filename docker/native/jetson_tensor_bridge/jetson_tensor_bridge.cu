#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

#pragma push_macro("__noinline__")
#undef __noinline__
#include <dlfcn.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <nvbufsurface.h>
#pragma pop_macro("__noinline__")

#include <cstdarg>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>

namespace {

constexpr const char* kSinkName = "rf_tensor_sink";
constexpr const char* kNvmmCapsFeature = "memory:NVMM";

enum DLDeviceType : int32_t {
    kDLCUDA = 2,
};

enum DLDataTypeCode : uint8_t {
    kDLUInt = 1,
};

struct DLDevice {
    DLDeviceType device_type;
    int32_t device_id;
};

struct DLDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};

struct DLTensor {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
};

struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor* self);
};

struct RfFrameInfo {
    uint32_t width;
    uint32_t height;
    int32_t fps_numerator;
    int32_t fps_denominator;
    int64_t duration_ns;
};

struct RfBridgeStats {
    uint64_t frames;
    uint64_t descriptor_maps;
    // The next four counters must stay zero on the zero-copy path; the
    // verify scripts assert on them to prove no host fallback executed.
    uint64_t host_pixel_maps;
    uint64_t host_to_device_copies;
    uint64_t device_to_host_copies;
    uint64_t array_flatten_copies;
    uint64_t conversion_kernels;
    uint64_t nvmm_frames;
    int32_t last_nvbuf_memory_type;
    int32_t last_egl_frame_type;
    int32_t last_egl_color_format;
};

struct RfTensorContext {
    void* allocation = nullptr;
    int device_id = 0;
    int64_t shape[3] = {0, 0, 0};
    DLManagedTensor managed{};
};

struct RfJetsonPipeline {
    GstElement* pipeline = nullptr;
    GstAppSink* sink = nullptr;
    GstSample* sample = nullptr;
    cudaStream_t stream = nullptr;
    int device_id = 0;
    RfBridgeStats stats{};
    std::atomic<bool> interrupted{false};
    std::mutex mutex;
};

using NvBufSurfaceMapEglImageFn = int (*)(NvBufSurface*, int);
using NvBufSurfaceUnMapEglImageFn = int (*)(NvBufSurface*, int);

struct NvBufSurfaceApi {
    void* library = nullptr;
    NvBufSurfaceMapEglImageFn map_egl_image = nullptr;
    NvBufSurfaceUnMapEglImageFn unmap_egl_image = nullptr;
    const char* error = nullptr;
};

struct ChannelMap {
    int red;
    int green;
    int blue;
};

std::once_flag g_gstreamer_once;
std::once_flag g_nvbufsurface_once;
NvBufSurfaceApi g_nvbufsurface;
char g_dlerror_message[256];

void write_error(char* destination, size_t capacity, const char* format, ...) {
    if (destination == nullptr || capacity == 0) {
        return;
    }
    va_list args;
    va_start(args, format);
    std::vsnprintf(destination, capacity, format, args);
    va_end(args);
    destination[capacity - 1] = '\0';
}

void initialize_gstreamer() {
    gst_init(nullptr, nullptr);
}

const char* capture_dlerror() {
    // dlerror() returns a pointer into a thread-local buffer that later dl*
    // calls (Python's ctypes dlopens constantly) overwrite; snapshot it.
    const char* message = dlerror();
    if (message == nullptr) {
        return nullptr;
    }
    std::snprintf(g_dlerror_message, sizeof(g_dlerror_message), "%s", message);
    return g_dlerror_message;
}

void initialize_nvbufsurface() {
    const char* candidates[] = {
        "libnvbufsurface.so.1.0.0",
        "libnvbufsurface.so",
        "/usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so.1.0.0",
        "/usr/lib/aarch64-linux-gnu/tegra/libnvbufsurface.so.1.0.0",
    };
    for (const char* candidate : candidates) {
        g_nvbufsurface.library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if (g_nvbufsurface.library != nullptr) {
            break;
        }
    }
    if (g_nvbufsurface.library == nullptr) {
        g_nvbufsurface.error = capture_dlerror();
        return;
    }
    g_nvbufsurface.map_egl_image =
        reinterpret_cast<NvBufSurfaceMapEglImageFn>(
            dlsym(g_nvbufsurface.library, "NvBufSurfaceMapEglImage"));
    g_nvbufsurface.unmap_egl_image =
        reinterpret_cast<NvBufSurfaceUnMapEglImageFn>(
            dlsym(g_nvbufsurface.library, "NvBufSurfaceUnMapEglImage"));
    if (g_nvbufsurface.map_egl_image == nullptr ||
        g_nvbufsurface.unmap_egl_image == nullptr) {
        g_nvbufsurface.error = capture_dlerror();
    }
}

bool get_channel_map(CUeglColorFormat format, ChannelMap* result) {
    if (result == nullptr) {
        return false;
    }
    switch (format) {
        case CU_EGL_COLOR_FORMAT_ABGR:
            *result = {0, 1, 2};
            return true;
        case CU_EGL_COLOR_FORMAT_RGBA:
            *result = {3, 2, 1};
            return true;
        case CU_EGL_COLOR_FORMAT_ARGB:
            *result = {2, 1, 0};
            return true;
        case CU_EGL_COLOR_FORMAT_BGRA:
            *result = {1, 2, 3};
            return true;
        default:
            return false;
    }
}

__device__ inline void store_rgb_chw(
    uchar4 pixel,
    ChannelMap channels,
    uint8_t* destination,
    uint32_t index,
    uint32_t plane_size) {
    const uint8_t values[4] = {pixel.x, pixel.y, pixel.z, pixel.w};
    destination[index] = values[channels.red];
    destination[plane_size + index] = values[channels.green];
    destination[2 * plane_size + index] = values[channels.blue];
}

__global__ void rgba_pitch_to_rgb_chw(
    const uint8_t* source,
    size_t source_pitch,
    uint8_t* destination,
    uint32_t width,
    uint32_t height,
    ChannelMap channels) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const uchar4* row = reinterpret_cast<const uchar4*>(source + y * source_pitch);
    const uint32_t index = y * width + x;
    store_rgb_chw(row[x], channels, destination, index, width * height);
}

__global__ void rgba_array_to_rgb_chw(
    cudaTextureObject_t texture,
    uint8_t* destination,
    uint32_t width,
    uint32_t height,
    ChannelMap channels) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const uchar4 pixel = tex2D<uchar4>(texture, x + 0.5f, y + 0.5f);
    const uint32_t index = y * width + x;
    store_rgb_chw(pixel, channels, destination, index, width * height);
}

void delete_managed_tensor(DLManagedTensor* managed) {
    if (managed == nullptr) {
        return;
    }
    auto* context = static_cast<RfTensorContext*>(managed->manager_ctx);
    if (context == nullptr) {
        return;
    }
    int previous_device = -1;
    cudaGetDevice(&previous_device);
    cudaSetDevice(context->device_id);
    if (context->allocation != nullptr) {
        cudaFree(context->allocation);
    }
    if (previous_device >= 0 && previous_device != context->device_id) {
        cudaSetDevice(previous_device);
    }
    delete context;
}

bool sample_has_nvmm_caps(GstSample* sample) {
    GstCaps* caps = gst_sample_get_caps(sample);
    if (caps == nullptr || gst_caps_is_empty(caps)) {
        return false;
    }
    for (guint index = 0; index < gst_caps_get_size(caps); ++index) {
        const GstCapsFeatures* features = gst_caps_get_features(caps, index);
        if (features != nullptr &&
            gst_caps_features_contains(features, kNvmmCapsFeature)) {
            return true;
        }
    }
    return false;
}

bool read_sample_info(GstSample* sample, RfFrameInfo* info) {
    if (sample == nullptr || info == nullptr) {
        return false;
    }
    GstCaps* caps = gst_sample_get_caps(sample);
    if (caps == nullptr || gst_caps_is_empty(caps)) {
        return false;
    }
    const GstStructure* structure = gst_caps_get_structure(caps, 0);
    int width = 0;
    int height = 0;
    int numerator = 0;
    int denominator = 1;
    if (!gst_structure_get_int(structure, "width", &width) ||
        !gst_structure_get_int(structure, "height", &height)) {
        return false;
    }
    gst_structure_get_fraction(
        structure, "framerate", &numerator, &denominator);
    info->width = static_cast<uint32_t>(width);
    info->height = static_cast<uint32_t>(height);
    info->fps_numerator = numerator;
    info->fps_denominator = denominator > 0 ? denominator : 1;
    return true;
}

bool read_bus_error(
    RfJetsonPipeline* handle,
    char* error,
    size_t error_capacity) {
    GstBus* bus = gst_element_get_bus(handle->pipeline);
    if (bus == nullptr) {
        return false;
    }
    GstMessage* message = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
    bool has_error = false;
    if (message != nullptr) {
        has_error = true;
        GError* gst_error = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &gst_error, &debug);
        write_error(
            error,
            error_capacity,
            "%s",
            gst_error != nullptr ? gst_error->message : "GStreamer pipeline error");
        if (gst_error != nullptr) {
            g_error_free(gst_error);
        }
        g_free(debug);
        gst_message_unref(message);
    }
    gst_object_unref(bus);
    return has_error;
}

bool pipeline_has_factory(RfJetsonPipeline* handle, const char* factory_name) {
    if (handle == nullptr || handle->pipeline == nullptr || factory_name == nullptr ||
        !GST_IS_BIN(handle->pipeline)) {
        return false;
    }
    GstIterator* iterator = gst_bin_iterate_recurse(GST_BIN(handle->pipeline));
    GValue item = G_VALUE_INIT;
    bool found = false;
    bool done = false;
    while (!done && !found) {
        switch (gst_iterator_next(iterator, &item)) {
            case GST_ITERATOR_OK: {
                auto* element = GST_ELEMENT(g_value_get_object(&item));
                GstElementFactory* factory = gst_element_get_factory(element);
                const gchar* name = factory == nullptr
                    ? nullptr
                    : gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
                found = name != nullptr && std::strcmp(name, factory_name) == 0;
                g_value_reset(&item);
                break;
            }
            case GST_ITERATOR_RESYNC:
                gst_iterator_resync(iterator);
                break;
            default:
                done = true;
                break;
        }
    }
    if (G_VALUE_TYPE(&item) != 0) {
        g_value_unset(&item);
    }
    gst_iterator_free(iterator);
    return found;
}

}  // namespace

extern "C" {

__attribute__((visibility("default")))
const char* rf_jetson_tensor_bridge_version() {
    return "3";
}

__attribute__((visibility("default")))
RfJetsonPipeline* rf_jetson_pipeline_create(
    const char* pipeline_description,
    int device_id,
    char* error,
    size_t error_capacity) {
    if (pipeline_description == nullptr || pipeline_description[0] == '\0') {
        write_error(error, error_capacity, "GStreamer pipeline is empty");
        return nullptr;
    }
    std::call_once(g_gstreamer_once, initialize_gstreamer);
    std::call_once(g_nvbufsurface_once, initialize_nvbufsurface);
    if (g_nvbufsurface.map_egl_image == nullptr ||
        g_nvbufsurface.unmap_egl_image == nullptr) {
        write_error(
            error,
            error_capacity,
            "NvBufSurface EGL API is unavailable: %s",
            g_nvbufsurface.error == nullptr ? "unknown error" : g_nvbufsurface.error);
        return nullptr;
    }
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaFree(nullptr);
    }
    if (cuda_status != cudaSuccess) {
        write_error(
            error,
            error_capacity,
            "CUDA device %d is unavailable: %s",
            device_id,
            cudaGetErrorString(cuda_status));
        return nullptr;
    }
    CUresult driver_status = cuInit(0);
    if (driver_status != CUDA_SUCCESS) {
        const char* driver_error = nullptr;
        cuGetErrorString(driver_status, &driver_error);
        write_error(
            error,
            error_capacity,
            "CUDA driver initialization failed: %s",
            driver_error == nullptr ? "unknown error" : driver_error);
        return nullptr;
    }

    GError* parse_error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_description, &parse_error);
    if (pipeline == nullptr || parse_error != nullptr) {
        write_error(
            error,
            error_capacity,
            "GStreamer pipeline parse failed: %s",
            parse_error == nullptr ? "unknown error" : parse_error->message);
        if (parse_error != nullptr) {
            g_error_free(parse_error);
        }
        if (pipeline != nullptr) {
            gst_object_unref(pipeline);
        }
        return nullptr;
    }
    if (!GST_IS_BIN(pipeline)) {
        write_error(error, error_capacity, "GStreamer pipeline is not a bin");
        gst_object_unref(pipeline);
        return nullptr;
    }
    GstElement* sink_element = gst_bin_get_by_name(GST_BIN(pipeline), kSinkName);
    if (sink_element == nullptr || !GST_IS_APP_SINK(sink_element)) {
        write_error(
            error,
            error_capacity,
            "GStreamer pipeline requires appsink name=%s",
            kSinkName);
        if (sink_element != nullptr) {
            gst_object_unref(sink_element);
        }
        gst_object_unref(pipeline);
        return nullptr;
    }

    auto* handle = new (std::nothrow) RfJetsonPipeline();
    if (handle == nullptr) {
        write_error(error, error_capacity, "Could not allocate pipeline state");
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        return nullptr;
    }
    handle->pipeline = pipeline;
    handle->sink = GST_APP_SINK(sink_element);
    handle->device_id = device_id;
    cuda_status = cudaStreamCreateWithFlags(&handle->stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
        write_error(
            error,
            error_capacity,
            "CUDA stream creation failed: %s",
            cudaGetErrorString(cuda_status));
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        delete handle;
        return nullptr;
    }
    const GstStateChangeReturn state_status =
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (state_status == GST_STATE_CHANGE_FAILURE) {
        write_error(error, error_capacity, "GStreamer could not enter PLAYING state");
        // A failed PLAYING transition can leave elements in READY/PAUSED with
        // running task threads; GStreamer refuses to dispose a non-NULL
        // pipeline, so reset it before dropping the reference.
        gst_element_set_state(pipeline, GST_STATE_NULL);
        cudaStreamDestroy(handle->stream);
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        delete handle;
        return nullptr;
    }
    return handle;
}

__attribute__((visibility("default")))
int rf_jetson_pipeline_grab(
    RfJetsonPipeline* handle,
    uint64_t timeout_ns,
    char* error,
    size_t error_capacity) {
    if (handle == nullptr) {
        write_error(error, error_capacity, "Pipeline handle is null");
        return -1;
    }
    if (handle->interrupted.load(std::memory_order_acquire)) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    if (handle->interrupted.load(std::memory_order_acquire)) {
        return 0;
    }
    if (handle->sample != nullptr) {
        gst_sample_unref(handle->sample);
        handle->sample = nullptr;
    }
    handle->sample = gst_app_sink_try_pull_sample(handle->sink, timeout_ns);
    if (handle->sample == nullptr) {
        if (handle->interrupted.load(std::memory_order_acquire) ||
            gst_app_sink_is_eos(handle->sink)) {
            return 0;
        }
        if (read_bus_error(handle, error, error_capacity)) {
            return -1;
        }
        // No frame, no error, no EOS: the finite timeout expired while the
        // stream is still live.
        return 2;
    }
    if (!sample_has_nvmm_caps(handle->sample)) {
        write_error(
            error,
            error_capacity,
            "GStreamer appsink frame is not memory:NVMM");
        gst_sample_unref(handle->sample);
        handle->sample = nullptr;
        return -1;
    }
    return 1;
}

__attribute__((visibility("default")))
int rf_jetson_pipeline_get_frame_info(
    RfJetsonPipeline* handle,
    RfFrameInfo* info,
    char* error,
    size_t error_capacity) {
    if (handle == nullptr || info == nullptr) {
        write_error(error, error_capacity, "Frame-info arguments are invalid");
        return -1;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    if (handle->sample == nullptr || !read_sample_info(handle->sample, info)) {
        write_error(error, error_capacity, "Frame caps do not contain dimensions");
        return -1;
    }
    gint64 duration = GST_CLOCK_TIME_NONE;
    if (gst_element_query_duration(handle->pipeline, GST_FORMAT_TIME, &duration)) {
        info->duration_ns = duration;
    } else {
        info->duration_ns = 0;
    }
    return 1;
}

__attribute__((visibility("default")))
int rf_jetson_pipeline_has_factory(
    RfJetsonPipeline* handle,
    const char* factory_name) {
    if (handle == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    return pipeline_has_factory(handle, factory_name) ? 1 : 0;
}

__attribute__((visibility("default")))
DLManagedTensor* rf_jetson_pipeline_retrieve(
    RfJetsonPipeline* handle,
    char* error,
    size_t error_capacity) {
    if (handle == nullptr) {
        write_error(error, error_capacity, "Pipeline handle is null");
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    if (handle->sample == nullptr) {
        write_error(error, error_capacity, "No grabbed frame is available");
        return nullptr;
    }
    // Bind the device context on the calling thread: retrieve() typically
    // runs on a consumer thread that has made no prior CUDA runtime call, and
    // the driver-API EGL registration below requires a current context.
    cudaError_t bind_status = cudaSetDevice(handle->device_id);
    if (bind_status == cudaSuccess) {
        bind_status = cudaFree(nullptr);
    }
    if (bind_status != cudaSuccess) {
        write_error(
            error,
            error_capacity,
            "CUDA device %d binding failed: %s",
            handle->device_id,
            cudaGetErrorString(bind_status));
        return nullptr;
    }

    GstSample* sample = handle->sample;
    handle->sample = nullptr;
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstMapInfo map = GST_MAP_INFO_INIT;
    bool buffer_mapped = false;
    bool egl_mapped = false;
    CUgraphicsResource graphics_resource = nullptr;
    cudaTextureObject_t texture = 0;
    RfTensorContext* tensor = nullptr;
    DLManagedTensor* result = nullptr;

    RfFrameInfo frame_info{};
    if (!read_sample_info(sample, &frame_info) || !sample_has_nvmm_caps(sample)) {
        write_error(error, error_capacity, "Frame caps are invalid");
        goto cleanup;
    }
    if (buffer == nullptr || !gst_buffer_map(buffer, &map, GST_MAP_READ) ||
        map.data == nullptr) {
        write_error(error, error_capacity, "Could not map the NvBufSurface descriptor");
        goto cleanup;
    }
    buffer_mapped = true;
    handle->stats.descriptor_maps += 1;

    {
        auto* surface = reinterpret_cast<NvBufSurface*>(map.data);
        handle->stats.last_nvbuf_memory_type = static_cast<int32_t>(surface->memType);
        if (surface->memType != NVBUF_MEM_SURFACE_ARRAY ||
            surface->surfaceList == nullptr || surface->batchSize == 0 ||
            surface->numFilled == 0) {
            write_error(
                error,
                error_capacity,
                "Frame is not a populated NvBufSurface SURFACE_ARRAY");
            goto cleanup;
        }
        if (surface->surfaceList[0].colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
            write_error(
                error,
                error_capacity,
                "NvBufSurface is not RGBA (format=%d)",
                static_cast<int>(surface->surfaceList[0].colorFormat));
            goto cleanup;
        }
        if (g_nvbufsurface.map_egl_image(surface, 0) != 0) {
            write_error(error, error_capacity, "NvBufSurface EGL mapping failed");
            goto cleanup;
        }
        egl_mapped = true;
        void* egl_image = surface->surfaceList[0].mappedAddr.eglImage;
        if (egl_image == nullptr) {
            write_error(error, error_capacity, "NvBufSurface EGL image is null");
            goto cleanup;
        }
        CUresult driver_status = cuGraphicsEGLRegisterImage(
            &graphics_resource,
            reinterpret_cast<EGLImageKHR>(egl_image),
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
        if (driver_status != CUDA_SUCCESS) {
            const char* driver_error = nullptr;
            cuGetErrorString(driver_status, &driver_error);
            write_error(
                error,
                error_capacity,
                "CUDA EGL registration failed: %s",
                driver_error == nullptr ? "unknown error" : driver_error);
            goto cleanup;
        }

        CUeglFrame egl_frame{};
        driver_status = cuGraphicsResourceGetMappedEglFrame(
            &egl_frame, graphics_resource, 0, 0);
        if (driver_status != CUDA_SUCCESS) {
            const char* driver_error = nullptr;
            cuGetErrorString(driver_status, &driver_error);
            write_error(
                error,
                error_capacity,
                "CUDA EGL frame mapping failed: %s",
                driver_error == nullptr ? "unknown error" : driver_error);
            goto cleanup;
        }
        handle->stats.last_egl_frame_type = static_cast<int32_t>(egl_frame.frameType);
        handle->stats.last_egl_color_format =
            static_cast<int32_t>(egl_frame.eglColorFormat);
        if (egl_frame.width < frame_info.width || egl_frame.height < frame_info.height ||
            egl_frame.planeCount != 1 || egl_frame.numChannels != 4 ||
            egl_frame.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8) {
            write_error(error, error_capacity, "CUDA EGL RGBA frame layout is invalid");
            goto cleanup;
        }

        ChannelMap channels{};
        if (!get_channel_map(egl_frame.eglColorFormat, &channels)) {
            write_error(
                error,
                error_capacity,
                "CUDA EGL color format is unsupported: %d",
                static_cast<int>(egl_frame.eglColorFormat));
            goto cleanup;
        }
        tensor = new (std::nothrow) RfTensorContext();
        if (tensor == nullptr) {
            write_error(error, error_capacity, "Could not allocate tensor state");
            goto cleanup;
        }
        tensor->device_id = handle->device_id;
        tensor->managed.manager_ctx = tensor;
        tensor->managed.deleter = delete_managed_tensor;
        const size_t output_size =
            static_cast<size_t>(frame_info.width) * frame_info.height * 3;
        cudaError_t cuda_status = cudaMalloc(&tensor->allocation, output_size);
        if (cuda_status != cudaSuccess) {
            write_error(
                error,
                error_capacity,
                "CUDA tensor allocation failed: %s",
                cudaGetErrorString(cuda_status));
            goto cleanup;
        }

        const dim3 threads(32, 8);
        const dim3 blocks(
            (frame_info.width + threads.x - 1) / threads.x,
            (frame_info.height + threads.y - 1) / threads.y);
        if (egl_frame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
            rgba_pitch_to_rgb_chw<<<blocks, threads, 0, handle->stream>>>(
                static_cast<const uint8_t*>(egl_frame.frame.pPitch[0]),
                egl_frame.pitch,
                static_cast<uint8_t*>(tensor->allocation),
                frame_info.width,
                frame_info.height,
                channels);
        } else if (egl_frame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
            cudaResourceDesc resource_description{};
            resource_description.resType = cudaResourceTypeArray;
            resource_description.res.array.array =
                reinterpret_cast<cudaArray_t>(egl_frame.frame.pArray[0]);
            cudaTextureDesc texture_description{};
            texture_description.addressMode[0] = cudaAddressModeClamp;
            texture_description.addressMode[1] = cudaAddressModeClamp;
            texture_description.filterMode = cudaFilterModePoint;
            texture_description.readMode = cudaReadModeElementType;
            texture_description.normalizedCoords = 0;
            cuda_status = cudaCreateTextureObject(
                &texture, &resource_description, &texture_description, nullptr);
            if (cuda_status != cudaSuccess) {
                write_error(
                    error,
                    error_capacity,
                    "CUDA texture creation failed: %s",
                    cudaGetErrorString(cuda_status));
                goto cleanup;
            }
            rgba_array_to_rgb_chw<<<blocks, threads, 0, handle->stream>>>(
                texture,
                static_cast<uint8_t*>(tensor->allocation),
                frame_info.width,
                frame_info.height,
                channels);
        } else {
            write_error(error, error_capacity, "CUDA EGL frame storage is unsupported");
            goto cleanup;
        }
        cuda_status = cudaGetLastError();
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaStreamSynchronize(handle->stream);
        }
        if (cuda_status != cudaSuccess) {
            write_error(
                error,
                error_capacity,
                "CUDA frame conversion failed: %s",
                cudaGetErrorString(cuda_status));
            goto cleanup;
        }

        tensor->shape[0] = 3;
        tensor->shape[1] = frame_info.height;
        tensor->shape[2] = frame_info.width;
        tensor->managed.dl_tensor.data = tensor->allocation;
        tensor->managed.dl_tensor.device = {kDLCUDA, handle->device_id};
        tensor->managed.dl_tensor.ndim = 3;
        tensor->managed.dl_tensor.dtype = {kDLUInt, 8, 1};
        tensor->managed.dl_tensor.shape = tensor->shape;
        tensor->managed.dl_tensor.strides = nullptr;
        tensor->managed.dl_tensor.byte_offset = 0;
        result = &tensor->managed;
        tensor = nullptr;
        handle->stats.frames += 1;
        handle->stats.conversion_kernels += 1;
        handle->stats.nvmm_frames += 1;
    }

cleanup:
    if (texture != 0) {
        cudaDestroyTextureObject(texture);
    }
    if (graphics_resource != nullptr) {
        cuGraphicsUnregisterResource(graphics_resource);
    }
    if (egl_mapped) {
        auto* surface = reinterpret_cast<NvBufSurface*>(map.data);
        g_nvbufsurface.unmap_egl_image(surface, 0);
    }
    if (buffer_mapped) {
        gst_buffer_unmap(buffer, &map);
    }
    gst_sample_unref(sample);
    if (tensor != nullptr) {
        delete_managed_tensor(&tensor->managed);
    }
    return result;
}

__attribute__((visibility("default")))
int rf_jetson_pipeline_get_stats(
    RfJetsonPipeline* handle,
    RfBridgeStats* stats) {
    if (handle == nullptr || stats == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    *stats = handle->stats;
    return 1;
}

__attribute__((visibility("default")))
void rf_jetson_dlpack_delete(DLManagedTensor* tensor) {
    if (tensor != nullptr && tensor->deleter != nullptr) {
        tensor->deleter(tensor);
    }
}

__attribute__((visibility("default")))
int rf_jetson_pipeline_interrupt(RfJetsonPipeline* handle) {
    if (handle == nullptr) {
        return -1;
    }
    handle->interrupted.store(true, std::memory_order_release);
    if (handle->sink != nullptr) {
        gst_app_sink_set_drop(handle->sink, TRUE);
        GstSample* queued_sample = nullptr;
        while ((queued_sample = gst_app_sink_try_pull_sample(handle->sink, 0)) !=
               nullptr) {
            gst_sample_unref(queued_sample);
        }
    }
    if (handle->pipeline != nullptr) {
        gst_element_set_state(handle->pipeline, GST_STATE_NULL);
    }
    return 1;
}

__attribute__((visibility("default")))
void rf_jetson_pipeline_release(RfJetsonPipeline* handle) {
    if (handle == nullptr) {
        return;
    }
    rf_jetson_pipeline_interrupt(handle);
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (handle->sample != nullptr) {
            gst_sample_unref(handle->sample);
            handle->sample = nullptr;
        }
        if (handle->sink != nullptr) {
            gst_app_sink_set_drop(handle->sink, TRUE);
            GstSample* queued_sample = nullptr;
            while ((queued_sample =
                        gst_app_sink_try_pull_sample(handle->sink, 0)) !=
                   nullptr) {
                gst_sample_unref(queued_sample);
            }
        }
        if (handle->pipeline != nullptr) {
            gst_element_set_state(handle->pipeline, GST_STATE_NULL);
        }
        if (handle->stream != nullptr) {
            cudaStreamDestroy(handle->stream);
        }
        if (handle->sink != nullptr) {
            gst_object_unref(handle->sink);
        }
        if (handle->pipeline != nullptr) {
            gst_object_unref(handle->pipeline);
        }
        // Null the element pointers before the handle is freed so a
        // contract-violating late interrupt() dereferences null instead of
        // freed objects. The Python wrapper serializes interrupt()/close().
        handle->stream = nullptr;
        handle->sink = nullptr;
        handle->pipeline = nullptr;
    }
    delete handle;
}

}  // extern "C"
