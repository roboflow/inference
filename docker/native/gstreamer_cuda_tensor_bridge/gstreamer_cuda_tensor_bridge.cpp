#include <cuda.h>
#include <gst/app/gstappsink.h>
#include <gst/cuda/gstcuda.h>
#include <gst/video/video.h>

#include <cstdarg>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>

namespace {

constexpr const char* kSinkName = "rf_tensor_sink";

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

struct RfCudaBridgeStats {
    uint64_t frames;
    uint64_t cuda_maps;
    uint64_t host_pixel_maps;
    uint64_t host_to_device_copies;
    uint64_t device_to_host_copies;
    uint64_t stream_synchronizations;
    uint64_t active_leases;
    int64_t last_channel_stride;
    int64_t last_row_stride;
};

struct RfLeaseCounter {
    std::atomic<uint64_t> references{1};
    std::atomic<uint64_t> active{0};
};

struct RfCudaTensorContext {
    GstSample* sample = nullptr;
    GstMemory* memory = nullptr;
    GstMapInfo map = GST_MAP_INFO_INIT;
    bool mapped = false;
    CUdevice device = 0;
    bool primary_context_retained = false;
    RfLeaseCounter* leases = nullptr;
    int64_t shape[3] = {0, 0, 0};
    int64_t strides[3] = {0, 0, 0};
    DLManagedTensor managed{};
};

struct RfCudaPipeline {
    GstElement* pipeline = nullptr;
    GstAppSink* sink = nullptr;
    GstSample* sample = nullptr;
    GstCudaContext* cuda_context = nullptr;
    CUcontext primary_context = nullptr;
    CUdevice device = 0;
    int device_id = 0;
    RfCudaBridgeStats stats{};
    RfLeaseCounter* leases = nullptr;
    std::atomic<bool> interrupted{false};
    std::mutex mutex;
    std::mutex stats_mutex;
};

std::once_flag g_gstreamer_once;

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
    gst_cuda_memory_init_once();
}

void delete_managed_tensor(DLManagedTensor* managed) {
    if (managed == nullptr) {
        return;
    }
    auto* context = static_cast<RfCudaTensorContext*>(managed->manager_ctx);
    if (context == nullptr) {
        return;
    }
    if (context->mapped && context->memory != nullptr) {
        gst_memory_unmap(context->memory, &context->map);
    }
    if (context->sample != nullptr) {
        gst_sample_unref(context->sample);
    }
    if (context->primary_context_retained) {
        cuDevicePrimaryCtxRelease(context->device);
    }
    if (context->leases != nullptr) {
        context->leases->active.fetch_sub(1, std::memory_order_relaxed);
        if (context->leases->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete context->leases;
        }
    }
    delete context;
}

bool sample_has_cuda_caps(GstSample* sample) {
    GstCaps* caps = gst_sample_get_caps(sample);
    if (caps == nullptr || gst_caps_is_empty(caps)) {
        return false;
    }
    for (guint index = 0; index < gst_caps_get_size(caps); ++index) {
        const GstCapsFeatures* features = gst_caps_get_features(caps, index);
        if (features != nullptr && gst_caps_features_contains(
                features, GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY)) {
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
    GstVideoInfo video_info{};
    if (!gst_video_info_from_caps(&video_info, caps)) {
        return false;
    }
    info->width = GST_VIDEO_INFO_WIDTH(&video_info);
    info->height = GST_VIDEO_INFO_HEIGHT(&video_info);
    info->fps_numerator = GST_VIDEO_INFO_FPS_N(&video_info);
    info->fps_denominator = GST_VIDEO_INFO_FPS_D(&video_info);
    if (info->fps_denominator <= 0) {
        info->fps_denominator = 1;
    }
    return true;
}

void read_bus_error(
    RfCudaPipeline* handle,
    char* error,
    size_t error_capacity) {
    GstBus* bus = gst_element_get_bus(handle->pipeline);
    if (bus == nullptr) {
        write_error(error, error_capacity, "GStreamer did not produce a frame");
        return;
    }
    GstMessage* message = gst_bus_pop_filtered(
        bus,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (message != nullptr && GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
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
    } else if (message != nullptr) {
        write_error(error, error_capacity, "GStreamer reached end of stream");
    } else {
        write_error(error, error_capacity, "GStreamer did not produce a frame");
    }
    if (message != nullptr) {
        gst_message_unref(message);
    }
    gst_object_unref(bus);
}

bool pipeline_has_factory(RfCudaPipeline* handle, const char* factory_name) {
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
const char* rf_gstreamer_cuda_tensor_bridge_version() {
    return "2";
}

__attribute__((visibility("default")))
RfCudaPipeline* rf_gstreamer_cuda_pipeline_create(
    const char* pipeline_description,
    int device_id,
    char* error,
    size_t error_capacity) {
    if (pipeline_description == nullptr || pipeline_description[0] == '\0') {
        write_error(error, error_capacity, "GStreamer pipeline is empty");
        return nullptr;
    }
    std::call_once(g_gstreamer_once, initialize_gstreamer);

    CUresult cuda_status = cuInit(0);
    CUdevice device = 0;
    CUcontext primary_context = nullptr;
    if (cuda_status == CUDA_SUCCESS) {
        cuda_status = cuDeviceGet(&device, device_id);
    }
    if (cuda_status == CUDA_SUCCESS) {
        cuda_status = cuDevicePrimaryCtxRetain(&primary_context, device);
    }
    if (cuda_status != CUDA_SUCCESS) {
        const char* cuda_error = nullptr;
        cuGetErrorString(cuda_status, &cuda_error);
        write_error(
            error,
            error_capacity,
            "CUDA primary context is unavailable: %s",
            cuda_error == nullptr ? "unknown error" : cuda_error);
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
        cuDevicePrimaryCtxRelease(device);
        return nullptr;
    }
    if (!GST_IS_BIN(pipeline)) {
        write_error(error, error_capacity, "GStreamer pipeline is not a bin");
        gst_object_unref(pipeline);
        cuDevicePrimaryCtxRelease(device);
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
        cuDevicePrimaryCtxRelease(device);
        return nullptr;
    }

    GstCudaContext* cuda_context =
        gst_cuda_context_new_wrapped(primary_context, device);
    if (cuda_context == nullptr) {
        write_error(error, error_capacity, "Could not wrap the CUDA primary context");
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        cuDevicePrimaryCtxRelease(device);
        return nullptr;
    }
    GstContext* gst_context = gst_context_new_cuda_context(cuda_context);
    gst_element_set_context(pipeline, gst_context);
    gst_context_unref(gst_context);

    auto* handle = new (std::nothrow) RfCudaPipeline();
    if (handle == nullptr) {
        write_error(error, error_capacity, "Could not allocate pipeline state");
        gst_object_unref(cuda_context);
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        cuDevicePrimaryCtxRelease(device);
        return nullptr;
    }
    handle->pipeline = pipeline;
    handle->sink = GST_APP_SINK(sink_element);
    handle->cuda_context = cuda_context;
    handle->primary_context = primary_context;
    handle->device = device;
    handle->device_id = device_id;
    handle->leases = new (std::nothrow) RfLeaseCounter();
    if (handle->leases == nullptr) {
        write_error(error, error_capacity, "Could not allocate tensor lease state");
        gst_object_unref(cuda_context);
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        cuDevicePrimaryCtxRelease(device);
        delete handle;
        return nullptr;
    }

    const GstStateChangeReturn state_status =
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (state_status == GST_STATE_CHANGE_FAILURE) {
        write_error(error, error_capacity, "GStreamer could not enter PLAYING state");
        gst_object_unref(cuda_context);
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        cuDevicePrimaryCtxRelease(device);
        delete handle->leases;
        delete handle;
        return nullptr;
    }
    return handle;
}

__attribute__((visibility("default")))
int rf_gstreamer_cuda_pipeline_grab(
    RfCudaPipeline* handle,
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
        if (handle->interrupted.load(std::memory_order_acquire)) {
            return 0;
        }
        read_bus_error(handle, error, error_capacity);
        return gst_app_sink_is_eos(handle->sink) ? 0 : -1;
    }
    if (!sample_has_cuda_caps(handle->sample)) {
        write_error(
            error,
            error_capacity,
            "GStreamer appsink frame is not memory:CUDAMemory");
        gst_sample_unref(handle->sample);
        handle->sample = nullptr;
        return -1;
    }
    return 1;
}

__attribute__((visibility("default")))
int rf_gstreamer_cuda_pipeline_get_frame_info(
    RfCudaPipeline* handle,
    RfFrameInfo* info,
    char* error,
    size_t error_capacity) {
    if (handle == nullptr || info == nullptr) {
        write_error(error, error_capacity, "Frame-info arguments are invalid");
        return -1;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    if (handle->sample == nullptr || !read_sample_info(handle->sample, info)) {
        write_error(error, error_capacity, "Frame caps do not contain video metadata");
        return -1;
    }
    gint64 duration = GST_CLOCK_TIME_NONE;
    info->duration_ns = gst_element_query_duration(
        handle->pipeline, GST_FORMAT_TIME, &duration) ? duration : 0;
    return 1;
}

__attribute__((visibility("default")))
int rf_gstreamer_cuda_pipeline_has_factory(
    RfCudaPipeline* handle,
    const char* factory_name) {
    if (handle == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    return pipeline_has_factory(handle, factory_name) ? 1 : 0;
}

__attribute__((visibility("default")))
DLManagedTensor* rf_gstreamer_cuda_pipeline_retrieve(
    RfCudaPipeline* handle,
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

    GstSample* sample = handle->sample;
    handle->sample = nullptr;
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (buffer == nullptr || gst_buffer_n_memory(buffer) != 1) {
        write_error(error, error_capacity, "CUDA frame must contain one GstMemory");
        gst_sample_unref(sample);
        return nullptr;
    }
    GstMemory* memory = gst_buffer_peek_memory(buffer, 0);
    if (memory == nullptr || !gst_is_cuda_memory(memory)) {
        write_error(error, error_capacity, "Frame memory is not GstCudaMemory");
        gst_sample_unref(sample);
        return nullptr;
    }
    if (GST_MEMORY_FLAG_IS_SET(
            memory,
            static_cast<GstMemoryFlags>(GST_CUDA_MEMORY_TRANSFER_NEED_UPLOAD))) {
        write_error(error, error_capacity, "CUDA frame requires a host upload");
        gst_sample_unref(sample);
        return nullptr;
    }

    auto* cuda_memory = GST_CUDA_MEMORY_CAST(memory);
    CUcontext memory_context = reinterpret_cast<CUcontext>(
        gst_cuda_context_get_handle(cuda_memory->context));
    if (memory_context != handle->primary_context) {
        write_error(error, error_capacity, "CUDA frame uses a different context");
        gst_sample_unref(sample);
        return nullptr;
    }
    const GstVideoInfo* video_info = &cuda_memory->info;
    if (GST_VIDEO_INFO_FORMAT(video_info) != GST_VIDEO_FORMAT_RGBP ||
        GST_VIDEO_INFO_N_PLANES(video_info) != 3 ||
        GST_VIDEO_INFO_COMP_PSTRIDE(video_info, 0) != 1 ||
        GST_VIDEO_INFO_COMP_PSTRIDE(video_info, 1) != 1 ||
        GST_VIDEO_INFO_COMP_PSTRIDE(video_info, 2) != 1) {
        write_error(error, error_capacity, "CUDA frame is not planar uint8 RGB");
        gst_sample_unref(sample);
        return nullptr;
    }
    const int64_t row_stride = GST_VIDEO_INFO_PLANE_STRIDE(video_info, 0);
    const int64_t second_row_stride = GST_VIDEO_INFO_PLANE_STRIDE(video_info, 1);
    const int64_t third_row_stride = GST_VIDEO_INFO_PLANE_STRIDE(video_info, 2);
    const int64_t channel_stride =
        GST_VIDEO_INFO_PLANE_OFFSET(video_info, 1) -
        GST_VIDEO_INFO_PLANE_OFFSET(video_info, 0);
    const int64_t second_channel_stride =
        GST_VIDEO_INFO_PLANE_OFFSET(video_info, 2) -
        GST_VIDEO_INFO_PLANE_OFFSET(video_info, 1);
    if (row_stride <= 0 || row_stride != second_row_stride ||
        row_stride != third_row_stride || channel_stride <= 0 ||
        channel_stride != second_channel_stride) {
        write_error(error, error_capacity, "CUDA RGB plane strides are incompatible");
        gst_sample_unref(sample);
        return nullptr;
    }

    auto* tensor = new (std::nothrow) RfCudaTensorContext();
    if (tensor == nullptr) {
        write_error(error, error_capacity, "Could not allocate tensor lease");
        gst_sample_unref(sample);
        return nullptr;
    }
    tensor->sample = sample;
    tensor->memory = memory;
    tensor->device = handle->device;
    tensor->managed.manager_ctx = tensor;
    tensor->managed.deleter = delete_managed_tensor;
    CUresult cuda_status = cuDevicePrimaryCtxRetain(
        &memory_context, handle->device);
    if (cuda_status != CUDA_SUCCESS) {
        write_error(error, error_capacity, "Could not retain the CUDA primary context");
        delete_managed_tensor(&tensor->managed);
        return nullptr;
    }
    tensor->primary_context_retained = true;

    gst_cuda_memory_sync(cuda_memory);
    if (!gst_memory_map(
            memory,
            &tensor->map,
            static_cast<GstMapFlags>(GST_MAP_READ | GST_MAP_CUDA))) {
        write_error(error, error_capacity, "Could not map CUDA frame memory");
        delete_managed_tensor(&tensor->managed);
        return nullptr;
    }
    tensor->mapped = true;

    tensor->shape[0] = 3;
    tensor->shape[1] = GST_VIDEO_INFO_HEIGHT(video_info);
    tensor->shape[2] = GST_VIDEO_INFO_WIDTH(video_info);
    tensor->strides[0] = channel_stride;
    tensor->strides[1] = row_stride;
    tensor->strides[2] = 1;
    tensor->managed.dl_tensor.data = tensor->map.data;
    tensor->managed.dl_tensor.device = {kDLCUDA, handle->device_id};
    tensor->managed.dl_tensor.ndim = 3;
    tensor->managed.dl_tensor.dtype = {kDLUInt, 8, 1};
    tensor->managed.dl_tensor.shape = tensor->shape;
    tensor->managed.dl_tensor.strides = tensor->strides;
    tensor->managed.dl_tensor.byte_offset =
        GST_VIDEO_INFO_PLANE_OFFSET(video_info, 0);

    {
        std::lock_guard<std::mutex> stats_lock(handle->stats_mutex);
        handle->stats.frames += 1;
        handle->stats.cuda_maps += 1;
        handle->stats.stream_synchronizations += 1;
        handle->stats.last_channel_stride = channel_stride;
        handle->stats.last_row_stride = row_stride;
    }
    handle->leases->references.fetch_add(1, std::memory_order_relaxed);
    handle->leases->active.fetch_add(1, std::memory_order_relaxed);
    tensor->leases = handle->leases;
    return &tensor->managed;
}

__attribute__((visibility("default")))
int rf_gstreamer_cuda_pipeline_get_stats(
    RfCudaPipeline* handle,
    RfCudaBridgeStats* stats) {
    if (handle == nullptr || stats == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(handle->stats_mutex);
    *stats = handle->stats;
    stats->active_leases = handle->leases->active.load(std::memory_order_relaxed);
    return 1;
}

__attribute__((visibility("default")))
void rf_gstreamer_cuda_dlpack_delete(DLManagedTensor* tensor) {
    if (tensor != nullptr && tensor->deleter != nullptr) {
        tensor->deleter(tensor);
    }
}

__attribute__((visibility("default")))
int rf_gstreamer_cuda_pipeline_interrupt(RfCudaPipeline* handle) {
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
void rf_gstreamer_cuda_pipeline_release(RfCudaPipeline* handle) {
    if (handle == nullptr) {
        return;
    }
    rf_gstreamer_cuda_pipeline_interrupt(handle);
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
        if (handle->sink != nullptr) {
            gst_object_unref(handle->sink);
        }
        if (handle->pipeline != nullptr) {
            gst_object_unref(handle->pipeline);
        }
        if (handle->cuda_context != nullptr) {
            gst_object_unref(handle->cuda_context);
        }
        cuDevicePrimaryCtxRelease(handle->device);
        if (handle->leases->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete handle->leases;
        }
    }
    delete handle;
}

}  // extern "C"
