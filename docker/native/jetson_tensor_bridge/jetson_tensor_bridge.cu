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
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <vector>

namespace {

constexpr const char* kSinkName = "rf_tensor_sink";
constexpr const char* kNvmmCapsFeature = "memory:NVMM";
// Upper bound on device buffers the per-pipeline tensor pool recycles. The
// steady state holds only the frames a consumer keeps in flight (typically
// one or two); this cap only bounds a slow or bursty consumer. Buffers are
// allocated lazily on demand, so a small cap costs nothing until it is hit.
constexpr size_t kJetsonTensorPoolBuffers = 8;
// Handoff queue depth in lossless (file) mode: enough to absorb consumer
// jitter without noticeably delaying backpressure. Must stay below
// kJetsonTensorPoolBuffers or the pool would starve the converter.
constexpr size_t kLosslessHandoffCapacity = 4;

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
    // --- ABI v5: per-phase timing of the streaming-thread conversion (ns). ---
    // Under concurrent TRT/torch GPU load individual native calls in
    // convert_sample_to_tensor() intermittently stall for hundreds of ms;
    // these accumulators (total + observed max per phase) name the stalling
    // call. Totals include failed conversion attempts (phases that ran).
    uint64_t egl_map_ns;
    uint64_t egl_map_max_ns;
    uint64_t cuda_register_ns;
    uint64_t cuda_register_max_ns;
    uint64_t texture_create_ns;
    uint64_t texture_create_max_ns;
    uint64_t kernel_launch_ns;
    uint64_t kernel_launch_max_ns;
    uint64_t sync_ns;
    uint64_t sync_max_ns;
    uint64_t cleanup_ns;
    uint64_t cleanup_max_ns;
    // Distinct dmabuf fds (NvBufSurfaceParams.bufferDesc) seen on this
    // pipeline — the decoder capture-pool size. A small, stable set is the
    // prerequisite for caching EGL/CUDA registrations per pool surface.
    uint64_t unique_buffer_fds;
    // --- ABI v7: EGL registration cache effectiveness. Hits skip the
    // per-frame NvBufSurfaceMapEglImage/cuGraphicsEGLRegisterImage/texture
    // creation/unregister sequence whose process-global locks measured
    // 13-19 ms mean (max 97 ms) under GPU saturation. Expected hit rate
    // after warmup: >99% (decoder pools recycle ~11 stable fds).
    uint64_t egl_cache_hits;
    uint64_t egl_cache_misses;
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

// One cached EGL/CUDA registration for a decoder-pool surface, keyed by its
// dmabuf fd. `egl_image` doubles as the validity token: if the surface comes
// back with a different (or cleared) mappedAddr.eglImage, the fd was reused
// for a new surface and the entry is rebuilt.
struct RfEglCacheEntry {
    uint64_t buffer_desc = 0;
    void* egl_image = nullptr;
    CUgraphicsResource resource = nullptr;
    CUeglFrame frame{};
    cudaTextureObject_t textures[2] = {0, 0};
    uint32_t texture_count = 0;
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
    // thread and publishes the finished CUDA tensor here. grab() waits on
    // `frame_ready`; retrieve() pops the oldest tensor. Two modes:
    //  * live (lossless_handoff=false, capacity 1): a newer frame replaces an
    //    uncollected one (latest-wins, like appsink drop=true, but the drop
    //    happens AFTER decode so nothing stalls) - correct for live streams
    //    where a source cannot be paused;
    //  * lossless (lossless_handoff=true, capacity kLosslessHandoffCapacity):
    //    the callback BLOCKS on `handoff_space` while the queue is full,
    //    holding the appsink sample - appsink (drop=false) queues up, the
    //    non-leaky queue fills, the decoder and demuxer stall and filesrc
    //    pauses. Decode is demand-paced and no frame is ever dropped -
    //    required for every-frame video-file processing.
    std::condition_variable frame_ready;
    std::condition_variable handoff_space;
    std::deque<RfTensorContext*> ready_tensors;
    bool lossless_handoff = false;
    size_t handoff_capacity = 1;
    std::atomic<bool> eos{false};
    bool conversion_failed = false;
    char conversion_error[1024] = {0};
    RfFrameInfo last_frame_info{};
    bool frame_info_valid = false;
    // The advertised source properties are established from the first frame.
    // If later caps disagree, fail recoverably instead of returning tensors
    // whose dimensions no longer match cached workflow metadata.
    bool caps_changed = false;
    char caps_change_error[1024] = {0};
    // dmabuf fds observed on this pipeline (guarded by `mutex`); its size is
    // exported as stats.unique_buffer_fds. Decoder capture pools are small
    // (single digits), so an ordered set is fine.
    std::set<uint64_t> seen_buffer_fds;
    // EGL/CUDA registration cache keyed by dmabuf fd (the DeepStream
    // pattern): decoder pools recycle a small stable set of surfaces, so
    // the EGL image mapping, CUDA graphics registration and texture objects
    // survive across frames instead of paying the process-global-lock
    // sequence per frame. Touched only on the streaming thread inside
    // convert_sample_to_tensor(); release() tears it down after the
    // streaming thread is joined (GST_STATE_NULL).
    std::vector<RfEglCacheEntry> egl_cache;
};

RfEglCacheEntry* find_egl_cache_entry(
    RfJetsonPipeline* handle, uint64_t buffer_desc) {
    for (auto& entry : handle->egl_cache) {
        if (entry.buffer_desc == buffer_desc) {
            return &entry;
        }
    }
    return nullptr;
}

// Frees the CUDA-side resources of an entry (textures + graphics
// registration). The surface's EGL mapping itself is deliberately left in
// place: it belongs to the decoder-pool surface and dies with it.
void destroy_egl_cache_entry_resources(RfEglCacheEntry* entry) {
    for (uint32_t plane = 0; plane < entry->texture_count; ++plane) {
        if (entry->textures[plane] != 0) {
            cudaDestroyTextureObject(entry->textures[plane]);
        }
    }
    if (entry->resource != nullptr) {
        cuGraphicsUnregisterResource(entry->resource);
    }
    entry->resource = nullptr;
    entry->egl_image = nullptr;
    entry->textures[0] = 0;
    entry->textures[1] = 0;
    entry->texture_count = 0;
}

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

bool frame_caps_changed(const RfFrameInfo& previous, const RfFrameInfo& current) {
    // Some sources first negotiate an unknown 0/1 framerate and later refine
    // it without changing the actual image layout. Dimensions are the unsafe
    // change because cached workflow metadata and tensor shape then disagree;
    // frame-specific FPS remains available to callers without forcing a reset.
    return previous.width != current.width || previous.height != current.height;
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

// Per-attempt phase durations measured inside convert_sample_to_tensor(),
// accumulated into RfBridgeStats under the handle mutex by the caller (the
// conversion itself runs lock-free on the streaming thread).
struct RfPhaseTimings {
    uint64_t egl_map_ns = 0;
    uint64_t cuda_register_ns = 0;
    uint64_t texture_create_ns = 0;
    uint64_t kernel_launch_ns = 0;
    uint64_t sync_ns = 0;
    uint64_t cleanup_ns = 0;
    uint64_t buffer_fd = 0;
    bool buffer_fd_valid = false;
    uint32_t egl_cache_hit = 0;
    uint32_t egl_cache_miss = 0;
};

uint64_t monotonic_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

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
    RfPhaseTimings* timings,
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
    uint64_t phase_start = 0;
    bool egl_resources_cached = false;
    RfEglCacheEntry* egl_cache_entry = nullptr;

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
        // A traditional Jetson decoder exports an NVRM surface array, which
        // reaches CUDA through EGL. Thor's OpenRM decoder instead exports a
        // CUDA-device surface: its dataPtr is already a CUDA-addressable,
        // pitched image. Both are GPU-native and neither should fall back to
        // a CPU frame path.
        const bool direct_cuda_surface =
            surface->memType == NVBUF_MEM_CUDA_DEVICE;
        if ((surface->memType != NVBUF_MEM_SURFACE_ARRAY &&
             !direct_cuda_surface) ||
            surface->surfaceList == nullptr || surface->batchSize == 0) {
            write_error(
                error,
                error_capacity,
                "Frame is not a supported GPU NvBufSurface "
                "(memType=%d surfaceArray=%d cudaDevice=%d surfaceList=%p "
                "batchSize=%u numFilled=%u)",
                static_cast<int>(surface->memType),
                static_cast<int>(NVBUF_MEM_SURFACE_ARRAY),
                static_cast<int>(NVBUF_MEM_CUDA_DEVICE),
                static_cast<void*>(surface->surfaceList),
                surface->batchSize,
                surface->numFilled);
            goto cleanup;
        }
        // bufferDesc is the decoder dmabuf identity used by the EGL path.
        // CUDA-device surfaces bypass EGL registration, so do not fold their
        // backend-specific descriptor value into this diagnostic.
        if (!direct_cuda_surface) {
            timings->buffer_fd = surface->surfaceList[0].bufferDesc;
            timings->buffer_fd_valid = true;
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
        if (direct_cuda_surface) {
            const NvBufSurfaceParams& params = surface->surfaceList[0];
            const auto* source = static_cast<const uint8_t*>(params.dataPtr);
            const size_t source_pitch = params.pitch;
            if (source == nullptr || source_pitch == 0) {
                write_error(
                    error,
                    error_capacity,
                    "CUDA-device surface has no pitched data pointer "
                    "(data=%p pitch=%zu)",
                    static_cast<const void*>(source),
                    source_pitch);
                goto cleanup;
            }
            if (!is_rgba &&
                (params.planeParams.num_planes < 2 ||
                 params.planeParams.offset[1] == 0)) {
                write_error(
                    error,
                    error_capacity,
                    "CUDA-device NV12 surface has invalid plane metadata "
                    "(planes=%u uvOffset=%u)",
                    params.planeParams.num_planes,
                    params.planeParams.offset[1]);
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
            tensor->allocation =
                handle->tensor_pool->acquire(output_size, &tensor->pooled);
            if (tensor->allocation == nullptr) {
                write_error(
                    error,
                    error_capacity,
                    "CUDA tensor allocation failed for %zu bytes",
                    output_size);
                goto cleanup;
            }

            const dim3 threads(32, 8);
            const dim3 blocks(
                (frame_info.width + threads.x - 1) / threads.x,
                (frame_info.height + threads.y - 1) / threads.y);
            phase_start = monotonic_ns();
            if (is_rgba) {
                const ChannelMap rgba_channels{0, 1, 2};
                rgba_pitch_to_rgb_chw<<<blocks, threads, 0, handle->stream>>>(
                    source,
                    source_pitch,
                    static_cast<uint8_t*>(tensor->allocation),
                    frame_info.width,
                    frame_info.height,
                    rgba_channels);
            } else {
                nv12_pitch_to_rgb_chw<<<blocks, threads, 0, handle->stream>>>(
                    source,
                    source + params.planeParams.offset[1],
                    source_pitch,
                    static_cast<uint8_t*>(tensor->allocation),
                    frame_info.width,
                    frame_info.height,
                    yuv_coeffs);
            }
            timings->kernel_launch_ns = monotonic_ns() - phase_start;
            phase_start = monotonic_ns();
            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status == cudaSuccess) {
                cuda_status = cudaStreamSynchronize(handle->stream);
            }
            timings->sync_ns = monotonic_ns() - phase_start;
            if (cuda_status != cudaSuccess) {
                write_error(
                    error,
                    error_capacity,
                    "CUDA-device frame conversion failed: %s",
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
            goto cleanup;
        }
        egl_cache_entry = find_egl_cache_entry(handle, timings->buffer_fd);
        if (egl_cache_entry != nullptr) {
            void* current_egl_image = surface->surfaceList[0].mappedAddr.eglImage;
            if (current_egl_image != nullptr &&
                egl_cache_entry->egl_image == current_egl_image) {
                egl_resources_cached = true;
                timings->egl_cache_hit = 1;
            } else {
                // The fd was reused for a different surface (pool
                // reallocation) - free the stale registration and rebuild
                // into the same slot below.
                destroy_egl_cache_entry_resources(egl_cache_entry);
            }
        }
        CUeglFrame egl_frame{};
        if (egl_resources_cached) {
            egl_frame = egl_cache_entry->frame;
        } else {
            timings->egl_cache_miss = 1;
            phase_start = monotonic_ns();
            if (g_nvbufsurface.map_egl_image(surface, 0) != 0) {
                write_error(
                    error, error_capacity, "NvBufSurface EGL mapping failed");
                goto cleanup;
            }
            timings->egl_map_ns = monotonic_ns() - phase_start;
            egl_mapped = true;
            void* egl_image = surface->surfaceList[0].mappedAddr.eglImage;
            if (egl_image == nullptr) {
                write_error(
                    error, error_capacity, "NvBufSurface EGL image is null");
                goto cleanup;
            }
            phase_start = monotonic_ns();
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
            timings->cuda_register_ns = monotonic_ns() - phase_start;
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
            phase_start = monotonic_ns();
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
            timings->kernel_launch_ns = monotonic_ns() - phase_start;
        } else if (egl_frame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
            const uint32_t texture_count = is_rgba ? 1 : 2;
            if (egl_resources_cached) {
                textures[0] = egl_cache_entry->textures[0];
                textures[1] = egl_cache_entry->textures[1];
            }
            phase_start = monotonic_ns();
            for (uint32_t plane = 0;
                 !egl_resources_cached && plane < texture_count;
                 ++plane) {
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
            timings->texture_create_ns = monotonic_ns() - phase_start;
            phase_start = monotonic_ns();
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
            timings->kernel_launch_ns = monotonic_ns() - phase_start;
        } else {
            write_error(error, error_capacity, "CUDA EGL frame storage is unsupported");
            goto cleanup;
        }
        // Timed as one phase: the sync is where the streaming thread parks on
        // the GPU behind whatever else (TRT/torch) currently occupies it.
        phase_start = monotonic_ns();
        cuda_status = cudaGetLastError();
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaStreamSynchronize(handle->stream);
        }
        timings->sync_ns = monotonic_ns() - phase_start;
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
        if (!egl_resources_cached) {
            RfEglCacheEntry fresh_entry{};
            fresh_entry.buffer_desc = timings->buffer_fd;
            fresh_entry.egl_image = surface->surfaceList[0].mappedAddr.eglImage;
            fresh_entry.resource = graphics_resource;
            fresh_entry.frame = egl_frame;
            fresh_entry.textures[0] = textures[0];
            fresh_entry.textures[1] = textures[1];
            fresh_entry.texture_count =
                egl_frame.frameType == CU_EGL_FRAME_TYPE_ARRAY
                    ? (is_rgba ? 1u : 2u)
                    : 0u;
            if (egl_cache_entry != nullptr) {
                *egl_cache_entry = fresh_entry;
            } else {
                handle->egl_cache.push_back(fresh_entry);
            }
            // Ownership moved to the cache: neutralise the locals so the
            // cleanup section leaves the registration/textures alive, and
            // keep the surface EGL-mapped for reuse on the next frame.
            graphics_resource = nullptr;
            textures[0] = 0;
            textures[1] = 0;
            egl_mapped = false;
            egl_resources_cached = true;
        }
        *frame_info_out = frame_info;
        result = tensor;
        tensor = nullptr;
    }

cleanup:
    phase_start = monotonic_ns();
    if (!egl_resources_cached) {
        // On a cache hit `textures` alias the cached objects - they must
        // survive this frame (even a failed one) for reuse.
        for (cudaTextureObject_t texture : textures) {
            if (texture != 0) {
                cudaDestroyTextureObject(texture);
            }
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
    timings->cleanup_ns = monotonic_ns() - phase_start;
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
    if (handle->lossless_handoff) {
        // Lossless (file) mode: park BEFORE converting, holding the appsink
        // sample - that is what backs the queue up and stalls the decoder,
        // so decode is demand-paced instead of frames being overwritten.
        std::unique_lock<std::mutex> lock(handle->mutex);
        handle->handoff_space.wait(lock, [handle]() {
            return handle->ready_tensors.size() < handle->handoff_capacity ||
                   handle->interrupted.load(std::memory_order_acquire);
        });
        if (handle->interrupted.load(std::memory_order_acquire)) {
            lock.unlock();
            gst_sample_unref(sample);
            return GST_FLOW_OK;
        }
    }
    char error[1024] = {0};
    RfFrameInfo frame_info{};
    RfEglDiagnostics diagnostics{};
    RfPhaseTimings timings{};
    RfTensorContext* tensor = convert_sample_to_tensor(
        handle, sample, &frame_info, &diagnostics, &timings, error, sizeof(error));
    gst_sample_unref(sample);
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        handle->stats.descriptor_maps += 1;
        handle->stats.last_nvbuf_memory_type = diagnostics.memory_type;
        handle->stats.last_egl_frame_type = diagnostics.frame_type;
        handle->stats.last_egl_color_format = diagnostics.color_format;
        const auto accumulate_phase =
            [](uint64_t* total, uint64_t* max_value, uint64_t sample_ns) {
                *total += sample_ns;
                if (sample_ns > *max_value) {
                    *max_value = sample_ns;
                }
            };
        accumulate_phase(
            &handle->stats.egl_map_ns,
            &handle->stats.egl_map_max_ns,
            timings.egl_map_ns);
        accumulate_phase(
            &handle->stats.cuda_register_ns,
            &handle->stats.cuda_register_max_ns,
            timings.cuda_register_ns);
        accumulate_phase(
            &handle->stats.texture_create_ns,
            &handle->stats.texture_create_max_ns,
            timings.texture_create_ns);
        accumulate_phase(
            &handle->stats.kernel_launch_ns,
            &handle->stats.kernel_launch_max_ns,
            timings.kernel_launch_ns);
        accumulate_phase(
            &handle->stats.sync_ns,
            &handle->stats.sync_max_ns,
            timings.sync_ns);
        accumulate_phase(
            &handle->stats.cleanup_ns,
            &handle->stats.cleanup_max_ns,
            timings.cleanup_ns);
        if (timings.buffer_fd_valid &&
            handle->seen_buffer_fds.insert(timings.buffer_fd).second) {
            handle->stats.unique_buffer_fds =
                static_cast<uint64_t>(handle->seen_buffer_fds.size());
        }
        handle->stats.egl_cache_hits += timings.egl_cache_hit;
        handle->stats.egl_cache_misses += timings.egl_cache_miss;
        if (tensor == nullptr) {
            handle->conversion_failed = true;
            std::snprintf(
                handle->conversion_error,
                sizeof(handle->conversion_error),
                "%s",
                error);
        } else {
            if (handle->frame_info_valid &&
                frame_caps_changed(handle->last_frame_info, frame_info)) {
                // The negotiated dimensions were established from the first
                // frame; a mid-stream change would hand the consumer tensors
                // whose shape no longer matches cached source metadata.
                std::snprintf(
                    handle->caps_change_error,
                    sizeof(handle->caps_change_error),
                    "Jetson source dimensions changed from %ux%u to %ux%u; "
                    "restart the source to refresh workflow metadata",
                    handle->last_frame_info.width,
                    handle->last_frame_info.height,
                    frame_info.width,
                    frame_info.height);
                handle->caps_changed = true;
                delete_managed_tensor(&tensor->managed);
            } else {
                if (!handle->lossless_handoff && !handle->ready_tensors.empty()) {
                    // Latest-wins (live mode): the consumer never collected the
                    // previous frame. Its buffer goes straight back to the pool.
                    // Never taken in lossless mode: only this streaming thread
                    // pushes, so the space awaited above cannot vanish.
                    delete_managed_tensor(&handle->ready_tensors.front()->managed);
                    handle->ready_tensors.pop_front();
                    handle->stats.frames_dropped_by_consumer += 1;
                }
                handle->ready_tensors.push_back(tensor);
                handle->last_frame_info = frame_info;
                handle->frame_info_valid = true;
                handle->stats.frames += 1;
                handle->stats.conversion_kernels += 1;
                handle->stats.nvmm_frames += 1;
            }
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
    // v7: per-fd EGL/CUDA registration cache (map/register/texture objects
    // survive across frames; the per-frame global-lock sequence is paid only
    // on pool warmup) + egl_cache_hits/egl_cache_misses appended to
    // RfBridgeStats (ABI change — python mirror must match).
    // v6: lossless (file) handoff mode — rf_jetson_pipeline_create() gained
    // the lossless_handoff parameter; bounded blocking FIFO replaces the
    // latest-wins slot for non-live sources so every file frame is served,
    // live behavior unchanged.
    // v5: per-phase conversion timing (egl_map/cuda_register/texture_create/
    // kernel_launch/sync/cleanup, total+max ns each) and unique_buffer_fds
    // appended to RfBridgeStats.
    // v4: streaming-thread conversion + tensor handoff (jetson-utils consume
    // model), direct NV12 path, frames_dropped_by_consumer added to
    // RfBridgeStats.
    return "7";
}

__attribute__((visibility("default")))
RfJetsonPipeline* rf_jetson_pipeline_create(
    const char* pipeline_description,
    int device_id,
    int lossless_handoff,
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
    handle->lossless_handoff = lossless_handoff != 0;
    handle->handoff_capacity =
        handle->lossless_handoff ? kLosslessHandoffCapacity : 1;
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
            return !handle->ready_tensors.empty() || handle->conversion_failed ||
                   handle->caps_changed ||
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
        if (handle->caps_changed) {
            write_error(error, error_capacity, "%s", handle->caps_change_error);
            handle->caps_changed = false;
            return -1;
        }
        if (!handle->ready_tensors.empty()) {
            // In lossless mode queued frames are served BEFORE EOS is
            // reported, so the tail of a video file is never lost.
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
        // The final frame and the EOS flag are published by the same
        // streaming thread, but this thread released the mutex before the
        // check above: a frame pushed in that window would be silently
        // bypassed by the EOS report. Re-check the queue under the lock so
        // end-of-stream is only reported once every converted frame has been
        // served (a lossless file tail must never be lost).
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (!handle->ready_tensors.empty()) {
            return 1;
        }
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
    RfTensorContext* tensor = nullptr;
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (handle->ready_tensors.empty()) {
            write_error(error, error_capacity, "No grabbed frame is available");
            return nullptr;
        }
        // The tensor was fully converted on the streaming thread; hand it
        // over. No CUDA call happens on the consumer thread (the DLPack
        // deleter binds the device itself when the consumer eventually drops
        // the tensor).
        tensor = handle->ready_tensors.front();
        handle->ready_tensors.pop_front();
    }
    // Wake a lossless-mode streaming thread parked on a full handoff queue.
    handle->handoff_space.notify_all();
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
    // Wake a grab() parked on the handoff queue - and a lossless-mode
    // streaming thread parked on a full queue - so interrupt is prompt.
    handle->frame_ready.notify_all();
    handle->handoff_space.notify_all();
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
        while (!handle->ready_tensors.empty()) {
            // interrupt() already reached GST_STATE_NULL, so the streaming
            // thread is joined and no further callback can repopulate the
            // queue; return the uncollected frames' buffers to the pool.
            delete_managed_tensor(&handle->ready_tensors.front()->managed);
            handle->ready_tensors.pop_front();
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
        if (!handle->egl_cache.empty()) {
            // The streaming thread is joined (GST_STATE_NULL above), so the
            // cache is exclusively ours. Free the CUDA-side registrations;
            // the surfaces' EGL mappings die with the decoder pool.
            int previous_device = -1;
            cudaGetDevice(&previous_device);
            cudaSetDevice(handle->device_id);
            for (auto& entry : handle->egl_cache) {
                destroy_egl_cache_entry_resources(&entry);
            }
            handle->egl_cache.clear();
            if (previous_device >= 0 && previous_device != handle->device_id) {
                cudaSetDevice(previous_device);
            }
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
