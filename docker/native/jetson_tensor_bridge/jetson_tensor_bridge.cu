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
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace {

constexpr const char* kSinkName = "rf_tensor_sink";
constexpr const char* kNvmmCapsFeature = "memory:NVMM";
// Upper bound on device buffers the per-pipeline tensor pool recycles. The
// steady state holds only the frames a consumer keeps in flight (typically
// one or two); this cap only bounds a slow or bursty consumer. Buffers are
// allocated lazily on demand, so a small cap costs nothing until it is hit.
constexpr size_t kJetsonTensorPoolBuffers = 8;

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
    // Frames decoded+converted on the streaming thread but replaced by a newer
    // frame before the consumer collected them (latest-wins handoff slot).
    uint64_t frames_dropped_by_consumer;
    int32_t last_nvbuf_memory_type;
    int32_t last_egl_frame_type;
    int32_t last_egl_color_format;
};

// Recycles fixed-size device allocations so the hot retrieve()/free path never
// calls cudaMalloc/cudaFree. On the Jetson unified-memory allocator a per-frame
// cudaFree synchronizes the whole device, stalling the consumer that drops a
// tensor; returning the buffer to a free list instead makes release a pure CPU
// push. The pool is reference-counted (shared_ptr) so it outlives the pipeline
// whenever a consumer is still holding a tensor.
class RfBufferPool {
 public:
    RfBufferPool(int device_id, size_t max_buffers)
        : device_id_(device_id), max_buffers_(max_buffers) {}

    ~RfBufferPool() {
        // The pool is only destroyed once the pipeline and every outstanding
        // tensor have released, so all pooled buffers are back on the free
        // list here. Bind the device because the destructor may run on a
        // consumer thread that never touched CUDA.
        int previous_device = -1;
        cudaGetDevice(&previous_device);
        cudaSetDevice(device_id_);
        for (void* buffer : free_list_) {
            cudaFree(buffer);
        }
        if (previous_device >= 0 && previous_device != device_id_) {
            cudaSetDevice(previous_device);
        }
    }

    // Hands out a device buffer of `size` bytes. The caller (retrieve()) has
    // already bound the device. `*pooled` reports whether release() should
    // recycle the buffer (true) or free it immediately (false, for one-off
    // over-budget or off-size allocations). Returns nullptr on cudaMalloc
    // failure.
    void* acquire(size_t size, bool* pooled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_size_ == 0) {
            buffer_size_ = size;  // Adopt the first frame's size as the pool size.
        }
        if (size == buffer_size_ && !free_list_.empty()) {
            void* buffer = free_list_.back();
            free_list_.pop_back();
            *pooled = true;
            return buffer;
        }
        // Cold start or pool miss: allocate. cudaMalloc under the lock is fine
        // here — this is the slow path, not the steady-state recycle.
        void* buffer = nullptr;
        if (cudaMalloc(&buffer, size) != cudaSuccess) {
            return nullptr;
        }
        // Only track pool-sized buffers within budget; anything else is a
        // one-off (resolution change mid-stream, or a consumer holding more
        // frames than the cap) that release() frees directly.
        if (size == buffer_size_ && total_pooled_ < max_buffers_) {
            total_pooled_ += 1;
            *pooled = true;
        } else {
            *pooled = false;
        }
        return buffer;
    }

    // Returns a buffer handed out by acquire(). Pooled buffers go back on the
    // free list (no CUDA call, hence no device-wide sync); one-offs are freed.
    // For the one-off path the caller must have bound the device.
    void release(void* buffer, bool pooled) {
        if (buffer == nullptr) {
            return;
        }
        if (pooled) {
            std::lock_guard<std::mutex> lock(mutex_);
            free_list_.push_back(buffer);
            return;
        }
        cudaFree(buffer);
    }

 private:
    int device_id_;
    size_t max_buffers_;
    size_t buffer_size_ = 0;
    size_t total_pooled_ = 0;
    std::vector<void*> free_list_;
    std::mutex mutex_;
};

struct RfTensorContext {
    void* allocation = nullptr;
    int device_id = 0;
    int64_t shape[3] = {0, 0, 0};
    // Non-null when `allocation` came from the pool; release routes back to it
    // instead of cudaFree. Held by shared_ptr so the pool survives a pipeline
    // that closes while this tensor is still alive.
    std::shared_ptr<RfBufferPool> pool;
    bool pooled = false;
    DLManagedTensor managed{};
};

struct RfJetsonPipeline {
    GstElement* pipeline = nullptr;
    GstAppSink* sink = nullptr;
    cudaStream_t stream = nullptr;
    int device_id = 0;
    std::shared_ptr<RfBufferPool> tensor_pool;
    RfBridgeStats stats{};
    std::atomic<bool> interrupted{false};
    std::mutex mutex;
    // Streaming-thread handoff (the jetson-utils consume model): the appsink
    // new-sample callback converts each frame on the GStreamer streaming
    // thread, releases the GStreamer buffer immediately (the decoder pool is
    // never held hostage by a slow consumer), and publishes the finished CUDA
    // tensor here. grab() waits on `frame_ready`; retrieve() takes the tensor.
    // A newer frame replaces an uncollected one (latest-wins, like appsink
    // drop=true, but the drop happens AFTER decode so nothing stalls).
    std::condition_variable frame_ready;
    RfTensorContext* ready_tensor = nullptr;
    std::atomic<bool> eos{false};
    bool conversion_failed = false;
    char conversion_error[1024] = {0};
    RfFrameInfo last_frame_info{};
    bool frame_info_valid = false;
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

// YUV -> RGB conversion coefficients for the direct-NV12 path (no nvvidconv
// RGBA hop): R = y_scale*(Y - y_offset) + r_v*(V-128), etc.
struct YuvCoeffs {
    float y_scale;
    float y_offset;
    float r_v;
    float g_u;
    float g_v;
    float b_u;
};

bool select_nv12_coeffs(NvBufSurfaceColorFormat format, YuvCoeffs* result) {
    switch (format) {
        case NVBUF_COLOR_FORMAT_NV12:  // BT.601 limited range (decoder default)
            *result = {1.1644f, 16.0f, 1.5960f, -0.3918f, -0.8130f, 2.0172f};
            return true;
        case NVBUF_COLOR_FORMAT_NV12_ER:  // BT.601 full range
            *result = {1.0f, 0.0f, 1.4020f, -0.3441f, -0.7141f, 1.7720f};
            return true;
        case NVBUF_COLOR_FORMAT_NV12_709:  // BT.709 limited range
            *result = {1.1644f, 16.0f, 1.7927f, -0.2132f, -0.5329f, 2.1124f};
            return true;
        case NVBUF_COLOR_FORMAT_NV12_709_ER:  // BT.709 full range
            *result = {1.0f, 0.0f, 1.5748f, -0.1873f, -0.4681f, 1.8556f};
            return true;
        default:
            return false;
    }
}

__device__ inline void store_yuv_as_rgb_chw(
    float y_value,
    float u_value,
    float v_value,
    YuvCoeffs coeffs,
    uint8_t* destination,
    uint32_t index,
    uint32_t plane_size) {
    const float luma = coeffs.y_scale * (y_value - coeffs.y_offset);
    const float d = u_value - 128.0f;
    const float e = v_value - 128.0f;
    const float red = luma + coeffs.r_v * e;
    const float green = luma + coeffs.g_u * d + coeffs.g_v * e;
    const float blue = luma + coeffs.b_u * d;
    destination[index] =
        static_cast<uint8_t>(fminf(fmaxf(red, 0.0f), 255.0f));
    destination[plane_size + index] =
        static_cast<uint8_t>(fminf(fmaxf(green, 0.0f), 255.0f));
    destination[2 * plane_size + index] =
        static_cast<uint8_t>(fminf(fmaxf(blue, 0.0f), 255.0f));
}

__global__ void nv12_pitch_to_rgb_chw(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    size_t pitch,
    uint8_t* destination,
    uint32_t width,
    uint32_t height,
    YuvCoeffs coeffs) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const float y_value = static_cast<float>(y_plane[y * pitch + x]);
    const uint8_t* uv_row = uv_plane + (y / 2) * pitch;
    const uint32_t uv_x = (x / 2) * 2;
    const float u_value = static_cast<float>(uv_row[uv_x]);
    const float v_value = static_cast<float>(uv_row[uv_x + 1]);
    const uint32_t index = y * width + x;
    store_yuv_as_rgb_chw(
        y_value, u_value, v_value, coeffs, destination, index, width * height);
}

__global__ void nv12_array_to_rgb_chw(
    cudaTextureObject_t y_texture,
    cudaTextureObject_t uv_texture,
    uint8_t* destination,
    uint32_t width,
    uint32_t height,
    YuvCoeffs coeffs) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const float y_value =
        static_cast<float>(tex2D<uint8_t>(y_texture, x + 0.5f, y + 0.5f));
    const uchar2 uv =
        tex2D<uchar2>(uv_texture, (x / 2) + 0.5f, (y / 2) + 0.5f);
    const uint32_t index = y * width + x;
    store_yuv_as_rgb_chw(
        y_value,
        static_cast<float>(uv.x),
        static_cast<float>(uv.y),
        coeffs,
        destination,
        index,
        width * height);
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
        if (context->pool != nullptr) {
            // Pooled buffers return without a CUDA call; only one-offs hit
            // cudaFree here, and both need the device bound (above) in case
            // this runs on a consumer thread that never touched CUDA.
            context->pool->release(context->allocation, context->pooled);
        } else {
            cudaFree(context->allocation);
        }
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

struct RfEglDiagnostics {
    int32_t memory_type = -1;
    int32_t frame_type = -1;
    int32_t color_format = -1;
};

// Convert one appsink sample into a pooled CHW RGB CUDA tensor. Runs on the
// GStreamer STREAMING thread (the jetson-utils consume model): the sample is
// fully consumed here and the caller unrefs it immediately afterwards, so the
// decoder's capture pool is never held across consumer latency. Takes no lock
// — it only touches handle fields that are immutable after create()
// (device_id, stream, tensor_pool). Supports both the direct decoder output
// (NV12 semi-planar, the RTSP path) and nvvidconv output (RGBA, the
// CSI/v4l2/file paths).
RfTensorContext* convert_sample_to_tensor(
    RfJetsonPipeline* handle,
    GstSample* sample,
    RfFrameInfo* frame_info_out,
    RfEglDiagnostics* diagnostics,
    char* error,
    size_t error_capacity) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstMapInfo map = GST_MAP_INFO_INIT;
    bool buffer_mapped = false;
    bool egl_mapped = false;
    CUgraphicsResource graphics_resource = nullptr;
    cudaTextureObject_t textures[2] = {0, 0};
    RfTensorContext* tensor = nullptr;
    RfTensorContext* result = nullptr;

    // Bind the device on this streaming thread (it has made no prior CUDA
    // runtime call); the driver-API EGL registration below needs a context.
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

    {
        auto* surface = reinterpret_cast<NvBufSurface*>(map.data);
        diagnostics->memory_type = static_cast<int32_t>(surface->memType);
        // nvv4l2decoder's single-surface GStreamer buffers on Thor carry a
        // valid surfaceList[0] while leaving numFilled at zero.  `numFilled`
        // is batch bookkeeping, not a validity requirement for this
        // unbatched appsink handoff; rejecting it turns a usable NVMM frame
        // into a false CPU-fallback-worthy failure.
        if (surface->memType != NVBUF_MEM_SURFACE_ARRAY ||
            surface->surfaceList == nullptr || surface->batchSize == 0) {
            write_error(
                error,
                error_capacity,
                "Frame is not a populated NvBufSurface SURFACE_ARRAY");
            goto cleanup;
        }
        const NvBufSurfaceColorFormat surface_format =
            surface->surfaceList[0].colorFormat;
        const bool is_rgba = surface_format == NVBUF_COLOR_FORMAT_RGBA;
        YuvCoeffs yuv_coeffs{};
        if (!is_rgba && !select_nv12_coeffs(surface_format, &yuv_coeffs)) {
            write_error(
                error,
                error_capacity,
                "NvBufSurface format is unsupported (format=%d, expected RGBA "
                "or an NV12 variant)",
                static_cast<int>(surface_format));
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
        diagnostics->frame_type = static_cast<int32_t>(egl_frame.frameType);
        diagnostics->color_format = static_cast<int32_t>(egl_frame.eglColorFormat);
        const uint32_t expected_planes = is_rgba ? 1 : 2;
        if (egl_frame.width < frame_info.width ||
            egl_frame.height < frame_info.height ||
            egl_frame.planeCount != expected_planes ||
            egl_frame.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8 ||
            (is_rgba && egl_frame.numChannels != 4)) {
            write_error(error, error_capacity, "CUDA EGL frame layout is invalid");
            goto cleanup;
        }

        ChannelMap channels{};
        if (is_rgba && !get_channel_map(egl_frame.eglColorFormat, &channels)) {
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
        tensor->pool = handle->tensor_pool;
        tensor->managed.manager_ctx = tensor;
        tensor->managed.deleter = delete_managed_tensor;
        const size_t output_size =
            static_cast<size_t>(frame_info.width) * frame_info.height * 3;
        tensor->allocation = handle->tensor_pool->acquire(output_size, &tensor->pooled);
        if (tensor->allocation == nullptr) {
            write_error(
                error,
                error_capacity,
                "CUDA tensor allocation failed for %zu bytes",
                output_size);
            goto cleanup;
        }

        cudaError_t cuda_status = cudaSuccess;
        const dim3 threads(32, 8);
        const dim3 blocks(
            (frame_info.width + threads.x - 1) / threads.x,
            (frame_info.height + threads.y - 1) / threads.y);
        if (egl_frame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
            if (is_rgba) {
                rgba_pitch_to_rgb_chw<<<blocks, threads, 0, handle->stream>>>(
                    static_cast<const uint8_t*>(egl_frame.frame.pPitch[0]),
                    egl_frame.pitch,
                    static_cast<uint8_t*>(tensor->allocation),
                    frame_info.width,
                    frame_info.height,
                    channels);
            } else {
                nv12_pitch_to_rgb_chw<<<blocks, threads, 0, handle->stream>>>(
                    static_cast<const uint8_t*>(egl_frame.frame.pPitch[0]),
                    static_cast<const uint8_t*>(egl_frame.frame.pPitch[1]),
                    egl_frame.pitch,
                    static_cast<uint8_t*>(tensor->allocation),
                    frame_info.width,
                    frame_info.height,
                    yuv_coeffs);
            }
        } else if (egl_frame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
            const uint32_t texture_count = is_rgba ? 1 : 2;
            for (uint32_t plane = 0; plane < texture_count; ++plane) {
                cudaResourceDesc resource_description{};
                resource_description.resType = cudaResourceTypeArray;
                resource_description.res.array.array =
                    reinterpret_cast<cudaArray_t>(egl_frame.frame.pArray[plane]);
                cudaTextureDesc texture_description{};
                texture_description.addressMode[0] = cudaAddressModeClamp;
                texture_description.addressMode[1] = cudaAddressModeClamp;
                texture_description.filterMode = cudaFilterModePoint;
                texture_description.readMode = cudaReadModeElementType;
                texture_description.normalizedCoords = 0;
                cuda_status = cudaCreateTextureObject(
                    &textures[plane],
                    &resource_description,
                    &texture_description,
                    nullptr);
                if (cuda_status != cudaSuccess) {
                    write_error(
                        error,
                        error_capacity,
                        "CUDA texture creation failed: %s",
                        cudaGetErrorString(cuda_status));
                    goto cleanup;
                }
            }
            if (is_rgba) {
                rgba_array_to_rgb_chw<<<blocks, threads, 0, handle->stream>>>(
                    textures[0],
                    static_cast<uint8_t*>(tensor->allocation),
                    frame_info.width,
                    frame_info.height,
                    channels);
            } else {
                nv12_array_to_rgb_chw<<<blocks, threads, 0, handle->stream>>>(
                    textures[0],
                    textures[1],
                    static_cast<uint8_t*>(tensor->allocation),
                    frame_info.width,
                    frame_info.height,
                    yuv_coeffs);
            }
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
        *frame_info_out = frame_info;
        result = tensor;
        tensor = nullptr;
    }

cleanup:
    for (cudaTextureObject_t texture : textures) {
        if (texture != 0) {
            cudaDestroyTextureObject(texture);
        }
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
    if (tensor != nullptr) {
        delete_managed_tensor(&tensor->managed);
    }
    return result;
}

GstFlowReturn handle_new_sample(GstAppSink* sink, gpointer user_data) {
    auto* handle = static_cast<RfJetsonPipeline*>(user_data);
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (sample == nullptr) {
        return GST_FLOW_OK;
    }
    if (handle->interrupted.load(std::memory_order_acquire)) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    char error[1024] = {0};
    RfFrameInfo frame_info{};
    RfEglDiagnostics diagnostics{};
    RfTensorContext* tensor = convert_sample_to_tensor(
        handle, sample, &frame_info, &diagnostics, error, sizeof(error));
    gst_sample_unref(sample);
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        handle->stats.descriptor_maps += 1;
        handle->stats.last_nvbuf_memory_type = diagnostics.memory_type;
        handle->stats.last_egl_frame_type = diagnostics.frame_type;
        handle->stats.last_egl_color_format = diagnostics.color_format;
        if (tensor == nullptr) {
            handle->conversion_failed = true;
            std::snprintf(
                handle->conversion_error,
                sizeof(handle->conversion_error),
                "%s",
                error);
        } else {
            if (handle->ready_tensor != nullptr) {
                // Latest-wins: the consumer never collected the previous
                // frame. Its buffer goes straight back to the pool.
                delete_managed_tensor(&handle->ready_tensor->managed);
                handle->stats.frames_dropped_by_consumer += 1;
            }
            handle->ready_tensor = tensor;
            handle->last_frame_info = frame_info;
            handle->frame_info_valid = true;
            handle->stats.frames += 1;
            handle->stats.conversion_kernels += 1;
            handle->stats.nvmm_frames += 1;
        }
    }
    handle->frame_ready.notify_all();
    return GST_FLOW_OK;
}

GstFlowReturn handle_new_preroll(GstAppSink* sink, gpointer user_data) {
    // Live pipelines re-deliver the preroll buffer as the first sample once
    // PLAYING; consuming it here just keeps the sink from holding it.
    GstSample* sample = gst_app_sink_pull_preroll(sink);
    if (sample != nullptr) {
        gst_sample_unref(sample);
    }
    (void)user_data;
    return GST_FLOW_OK;
}

void handle_appsink_eos(GstAppSink* sink, gpointer user_data) {
    auto* handle = static_cast<RfJetsonPipeline*>(user_data);
    (void)sink;
    handle->eos.store(true, std::memory_order_release);
    handle->frame_ready.notify_all();
}

}  // namespace

extern "C" {

__attribute__((visibility("default")))
const char* rf_jetson_tensor_bridge_version() {
    // v4: streaming-thread conversion + tensor handoff (jetson-utils consume
    // model), direct NV12 path, frames_dropped_by_consumer added to
    // RfBridgeStats (ABI change — python mirror must match).
    return "4";
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
    handle->tensor_pool =
        std::make_shared<RfBufferPool>(device_id, kJetsonTensorPoolBuffers);
    if (handle->tensor_pool == nullptr) {
        write_error(error, error_capacity, "Could not allocate tensor buffer pool");
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        delete handle;
        return nullptr;
    }
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
    // Consume frames via appsink callbacks on the GStreamer streaming thread
    // (jetson-utils model): each sample is converted and released immediately,
    // so a slow consumer can never pin the decoder's capture pool or force
    // drops upstream. Must be installed before the PLAYING transition.
    GstAppSinkCallbacks sink_callbacks{};
    sink_callbacks.eos = handle_appsink_eos;
    sink_callbacks.new_preroll = handle_new_preroll;
    sink_callbacks.new_sample = handle_new_sample;
    gst_app_sink_set_callbacks(handle->sink, &sink_callbacks, handle, nullptr);
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
    {
        std::unique_lock<std::mutex> lock(handle->mutex);
        const auto frame_or_terminal = [handle]() {
            return handle->ready_tensor != nullptr || handle->conversion_failed ||
                   handle->interrupted.load(std::memory_order_acquire) ||
                   handle->eos.load(std::memory_order_acquire);
        };
        if (!frame_or_terminal()) {
            handle->frame_ready.wait_for(
                lock, std::chrono::nanoseconds(timeout_ns), frame_or_terminal);
        }
        if (handle->interrupted.load(std::memory_order_acquire)) {
            return 0;
        }
        if (handle->conversion_failed) {
            write_error(error, error_capacity, "%s", handle->conversion_error);
            handle->conversion_failed = false;
            return -1;
        }
        if (handle->ready_tensor != nullptr) {
            return 1;
        }
    }
    // Nothing ready. Check the bus BEFORE the EOS flag: a pipeline that died
    // during startup (RTSP connect/auth failure, autoplug failure) never
    // delivers a frame — the real error would otherwise be misclassified as a
    // silent end-of-stream. A genuine EOS posts no ERROR, so it still
    // returns 0.
    if (read_bus_error(handle, error, error_capacity)) {
        return -1;
    }
    if (handle->eos.load(std::memory_order_acquire) ||
        gst_app_sink_is_eos(handle->sink)) {
        return 0;
    }
    // No frame, no error, no EOS: the finite timeout expired while the
    // stream is still live.
    return 2;
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
    if (!handle->frame_info_valid) {
        write_error(error, error_capacity, "Frame caps do not contain dimensions");
        return -1;
    }
    *info = handle->last_frame_info;
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
    if (handle->ready_tensor == nullptr) {
        write_error(error, error_capacity, "No grabbed frame is available");
        return nullptr;
    }
    // The tensor was fully converted on the streaming thread; hand it over.
    // No CUDA call happens on the consumer thread (the DLPack deleter binds
    // the device itself when the consumer eventually drops the tensor).
    RfTensorContext* tensor = handle->ready_tensor;
    handle->ready_tensor = nullptr;
    return &tensor->managed;
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
    // Wake a grab() parked on the handoff slot so interrupt is prompt.
    handle->frame_ready.notify_all();
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
        if (handle->ready_tensor != nullptr) {
            // interrupt() already reached GST_STATE_NULL, so the streaming
            // thread is joined and no further callback can repopulate the
            // slot; return the uncollected frame's buffer to the pool.
            delete_managed_tensor(&handle->ready_tensor->managed);
            handle->ready_tensor = nullptr;
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
