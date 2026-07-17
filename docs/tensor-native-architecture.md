# Tensor-native video, detection, and tracking architecture

This document describes the production path added around Superiorvision,
Tracktors, and the hardware video consumers. It is deliberately a data-flow
description: the compatible public APIs remain, while numeric work stays on
the CUDA device until a caller explicitly asks for an external representation.

## One device-resident path

```mermaid
flowchart LR
    camera["Camera / file / RTSP\ncompressed video"] --> decode

    subgraph decode["Hardware video consumer"]
        gst["GStreamer / FFmpeg source"] --> hw["NVIDIA hardware decode\nNVMM or CUDAMemory"]
        hw --> bridge["Native bridge\nCUDA surface → DLPack"]
    end

    bridge --> frame["torch.Tensor\nCUDA • uint8 • CHW"]
    frame --> model["TensorRT / CUDA model"]
    model --> detections["Detection tensors\nxyxy • confidence • class • masks"]

    subgraph sv["Superiorvision — API-compatible tensor layer"]
        detections --> views["Detections / tensor views"]
        views --> geometry["IoU • box transforms • NMS/NMM\nzones • CompactMask"]
    end

    geometry --> tracktors
    subgraph tracktors["Tracktors — batched tracker execution"]
        executor["CUDABatchExecutor"] --> engines["SORT / ByteTrack whole-frame engines"]
        engines --> state["Persistent CUDA state arena"]
        state --> association["Batched IoU + assignment\nTriton Kalman predict/update"]
        association --> ids["CUDA tracker IDs + tracked tensors"]
    end

    ids --> workflow["Tensor-native Workflow blocks"]
    workflow --> boundary["Optional boundary only\nrender • serialization • external API"]

    classDef device fill:#173b5e,color:#fff,stroke:#46a2da,stroke-width:2px;
    classDef boundary fill:#5a3b17,color:#fff,stroke:#e7a73d,stroke-width:2px;
    class frame,detections,views,geometry,executor,engines,state,association,ids,workflow device;
    class boundary boundary;
```

The orange endpoint is intentional. Rendering, strings, ragged metadata,
dataset/video I/O, and serialization may need CPU-facing objects; numeric
frames, detections, association, Kalman state, and tracker IDs do not.

## Superiorvision: preserve the interface, replace the numeric substrate

```mermaid
flowchart TB
    api["Existing supervision-compatible API\nsv.Detections, trackers, zones, annotators"]

    api --> contract{"Numeric field?"}
    contract -->|"yes"| tensor["torch.Tensor on caller's device\nno implicit NumPy round trip"]
    contract -->|"ragged/object metadata"| metadata["Compatible Python/object metadata\nkept out of numeric hot path"]

    tensor --> ops
    subgraph ops["Tensor-native operations"]
        direction LR
        boxes["Tensor box views\nxyxy / xywh / centers"]
        overlap["Vectorized overlap\nIoU / NMS / NMM"]
        masks["CompactMask\npacked mask representation"]
        analytics["Zones and detection analytics"]
        boxes --> overlap
        boxes --> analytics
        masks --> overlap
    end

    ops --> compatible["Same result shape / public semantics"]
    metadata --> compatible
    compatible --> downstream["Inference workflows and Tracktors"]

    classDef gpu fill:#173b5e,color:#fff,stroke:#46a2da,stroke-width:2px;
    classDef compat fill:#374151,color:#fff,stroke:#9ca3af,stroke-width:2px;
    class tensor,boxes,overlap,masks,analytics gpu;
    class api,metadata,compatible compat;
```

The speedup is mostly not a new box formula: it removes repeated device↔host
conversion and lets all per-detection operations use vectorized Torch kernels
on tensors that the model already produced.

## Tracktors: from independent Python updates to one CUDA transaction

```mermaid
flowchart TB
    batch["Aligned Workflow Batch\nN streams of CUDA Detections"] --> bucket["CUDABatchExecutor\ngroups compatible tracker streams"]
    bucket --> fast{"Eligible SORT / ByteTrack\nCUDA cohort?"}
    fast -->|"yes"| arena
    fast -->|"not yet / exact edge case"| fallback["Exact public tracker update\ncompatibility fallback"]

    subgraph arena["Whole-frame CUDA fast path"]
        direction TB
        canonical["Canonical persistent state arena\nmeans • covariance • age • IDs"]
        packed["Packed / padded ragged detections\nand stream offsets"]
        canonical --> predict["Batched Kalman predict\nTriton"]
        packed --> iou["Segmented pairwise IoU"]
        predict --> iou
        iou --> assign["Exact segmented assignment\nGPU"]
        assign --> update["Batched Kalman update\nTriton"]
        update --> lifecycle["Device-resident matched-track state\nIDs remain CUDA tensors"]
    end

    lifecycle --> results["Per-stream Detections results\noriginal ordering restored"]
    fallback --> results

    note["Why it is faster:\n• one persistent executor\n• fewer launches / no Python worker fan-out\n• state is not repacked every frame\n• matrix math, association, and Kalman updates are batched"] -.-> arena

    classDef gpu fill:#173b5e,color:#fff,stroke:#46a2da,stroke-width:2px;
    classDef fallback fill:#5a3b17,color:#fff,stroke:#e7a73d,stroke-width:2px;
    class batch,bucket,canonical,packed,predict,iou,assign,update,lifecycle,results gpu;
    class fallback fallback;
```

The fast path preserves scalar Trackers semantics. Less regular lifecycle
events (mixed confidence, spawning, retirement, or unsupported combinations)
retain an exact fallback while their device-resident plans are expanded.

## Video consumers replacing jetson-utils

```mermaid
flowchart LR
    subgraph inputs["Input kinds"]
        rtsp["RTSP / RTSPS\nvideo track only"]
        csi["CSI camera"]
        v4l2["V4L2 camera"]
        file["File / URI"]
    end

    subgraph jetson["JetsonVideoFrameProducer — Jetson-native path"]
        rtspsrc["rtspsrc TCP\n→ RTP depay → parser"]
        argsrc["nvarguscamerasrc"]
        v4l2src["v4l2src"]
        uri["URI demux/decode"]
        decoder["nvv4l2decoder / nvjpegdec\nNVMM surface"]
        native["libroboflow_jetson_tensor\nCUDA NV12 → RGB CHW\nDLPack lease"]
        rtspsrc --> decoder
        argsrc --> native
        v4l2src --> decoder
        uri --> decoder
        decoder --> native
    end

    subgraph dgpu["GStreamerCudaFrameProducer — dGPU path"]
        cudauri["uridecodebin + NVIDIA decoder\nCUDAMemory"]
        scale["cudaconvertscale\nRGB planar"]
        cudabridge["libroboflow_gstreamer_cuda_tensor\nDLPack lease"]
        cudauri --> scale --> cudabridge
    end

    rtsp --> rtspsrc
    csi --> argsrc
    v4l2 --> v4l2src
    file --> uri
    file --> cudauri
    rtsp --> cudauri

    native --> tensor["CUDA uint8 CHW torch.Tensor"]
    cudabridge --> tensor
    tensor --> inference["Inference tensor workflow"]

    classDef device fill:#173b5e,color:#fff,stroke:#46a2da,stroke-width:2px;
    classDef source fill:#374151,color:#fff,stroke:#9ca3af,stroke-width:2px;
    class tensor,inference,decoder,native,cudauri,scale,cudabridge device;
    class rtsp,csi,v4l2,file source;
```

The explicit Jetson RTSP chain only links the camera's video track, uses TCP by
default, queues compressed data, decodes to NVMM, and performs the NV12→RGB
conversion in CUDA. That avoids jetson-utils, avoids OpenCV's CPU frame path,
and avoids an extra VIC conversion/buffer pool. The native bridges expose
timeouts, interruption, pipeline-factory checks, and counters for host pixel
maps, host/device copies, and CUDA synchronization so the zero-copy claim is
testable rather than aspirational.
