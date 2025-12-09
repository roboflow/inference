/**
 * WebRTC Video File Uploader
 *
 * Sample ES6 class demonstrating how to upload video files via WebRTC data channels
 * using the new binary chunked protocol.
 *
 * Protocol Specification:
 * -----------------------
 * Video Upload Header (8 bytes, little-endian):
 *   - chunk_index: uint32 (bytes 0-3)
 *   - total_chunks: uint32 (bytes 4-7)
 *   - payload: raw video bytes
 *
 * Output Message Header (12 bytes, little-endian):
 *   - frame_id: uint32 (bytes 0-3)
 *   - chunk_index: uint32 (bytes 4-7)
 *   - total_chunks: uint32 (bytes 8-11)
 *   - payload: JSON UTF-8 string
 */

class WebRTCVideoUploader {
  static CHUNK_SIZE = 48 * 1024; // 48KB - safe for all WebRTC implementations

  /**
   * @param {RTCPeerConnection} peerConnection - Existing WebRTC peer connection
   */
  constructor(peerConnection) {
    this.peerConnection = peerConnection;
    this.videoUploadChannel = null;
    this.outputReassembler = new Map(); // frame_id -> { chunks: Map, totalChunks: number }
    this.outputCallback = null;
  }

  /**
   * Creates the video_upload data channel for sending video files
   * @returns {RTCDataChannel}
   */
  createVideoUploadChannel() {
    this.videoUploadChannel = this.peerConnection.createDataChannel("video_upload", {
      ordered: true,
    });

    this.videoUploadChannel.binaryType = "arraybuffer";

    this.videoUploadChannel.onopen = () => {
      console.log("Video upload channel opened");
    };

    this.videoUploadChannel.onerror = (error) => {
      console.error("Video upload channel error:", error);
    };

    return this.videoUploadChannel;
  }

  /**
   * Sets up handler for incoming output messages on a data channel
   * @param {RTCDataChannel} dataChannel - The data channel to listen on
   */
  setupOutputHandler(dataChannel) {
    dataChannel.binaryType = "arraybuffer";

    dataChannel.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        this._handleBinaryOutput(event.data);
      } else if (typeof event.data === "string") {
        // Legacy JSON string format
        try {
          const output = JSON.parse(event.data);
          this._emitOutput(output);
        } catch (e) {
          console.error("Failed to parse output:", e);
        }
      }
    };
  }

  /**
   * Register callback for receiving processed outputs
   * @param {Function} callback - Called with parsed output data
   */
  onOutput(callback) {
    this.outputCallback = callback;
  }

  /**
   * Upload a video file via the data channel
   * @param {File} file - Video file to upload
   * @returns {Promise<void>} Resolves when upload is complete
   */
  async uploadVideoFile(file) {
    if (!this.videoUploadChannel) {
      throw new Error("Video upload channel not created. Call createVideoUploadChannel() first.");
    }

    if (this.videoUploadChannel.readyState !== "open") {
      await this._waitForChannelOpen(this.videoUploadChannel);
    }

    const arrayBuffer = await file.arrayBuffer();
    const totalSize = arrayBuffer.byteLength;
    const totalChunks = Math.ceil(totalSize / WebRTCVideoUploader.CHUNK_SIZE);

    console.log(`Uploading ${file.name}: ${totalSize} bytes in ${totalChunks} chunks`);

    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      const start = chunkIndex * WebRTCVideoUploader.CHUNK_SIZE;
      const end = Math.min(start + WebRTCVideoUploader.CHUNK_SIZE, totalSize);
      const chunkData = arrayBuffer.slice(start, end);

      const header = this._createVideoUploadHeader(chunkIndex, totalChunks);
      const message = this._concatBuffers(header, chunkData);

      // Wait if buffer is getting full
      await this._waitForBufferDrain(this.videoUploadChannel);

      this.videoUploadChannel.send(message);

      // Progress logging
      if ((chunkIndex + 1) % 100 === 0 || chunkIndex === totalChunks - 1) {
        console.log(`Upload progress: ${chunkIndex + 1}/${totalChunks} chunks`);
      }
    }

    console.log("Video upload complete");
  }

  /**
   * Creates the 8-byte binary header for video upload chunks
   * @param {number} chunkIndex - Current chunk index (0-based)
   * @param {number} totalChunks - Total number of chunks
   * @returns {ArrayBuffer} 8-byte header
   */
  _createVideoUploadHeader(chunkIndex, totalChunks) {
    const header = new ArrayBuffer(8);
    const view = new DataView(header);

    view.setUint32(0, chunkIndex, true); // little-endian
    view.setUint32(4, totalChunks, true);

    return header;
  }

  /**
   * Handles incoming binary output messages
   * @param {ArrayBuffer} data - Raw binary message
   */
  _handleBinaryOutput(data) {
    if (data.byteLength < 12) {
      console.error("Invalid output message: too short");
      return;
    }

    const view = new DataView(data);
    const frameId = view.getUint32(0, true);
    const chunkIndex = view.getUint32(4, true);
    const totalChunks = view.getUint32(8, true);
    const payload = data.slice(12);

    this._reassembleOutput(frameId, chunkIndex, totalChunks, payload);
  }

  /**
   * Reassembles chunked output messages
   * @param {number} frameId - Unique frame identifier
   * @param {number} chunkIndex - Current chunk index
   * @param {number} totalChunks - Total chunks for this frame
   * @param {ArrayBuffer} payload - Chunk payload
   */
  _reassembleOutput(frameId, chunkIndex, totalChunks, payload) {
    if (!this.outputReassembler.has(frameId)) {
      this.outputReassembler.set(frameId, {
        chunks: new Map(),
        totalChunks: totalChunks,
      });
    }

    const frameData = this.outputReassembler.get(frameId);
    frameData.chunks.set(chunkIndex, payload);

    // Check if all chunks received
    if (frameData.chunks.size === frameData.totalChunks) {
      // Reassemble in order
      const orderedChunks = [];
      for (let i = 0; i < frameData.totalChunks; i++) {
        orderedChunks.push(frameData.chunks.get(i));
      }

      const completePayload = this._concatBuffers(...orderedChunks);
      const jsonString = new TextDecoder().decode(completePayload);

      try {
        const output = JSON.parse(jsonString);
        this._emitOutput(output);
      } catch (e) {
        console.error("Failed to parse reassembled output:", e);
      }

      // Clean up
      this.outputReassembler.delete(frameId);
    }
  }

  /**
   * Emits parsed output to registered callback
   * @param {Object} output - Parsed output object
   */
  _emitOutput(output) {
    if (this.outputCallback) {
      this.outputCallback(output);
    }
  }

  /**
   * Concatenates multiple ArrayBuffers into one
   * @param {...ArrayBuffer} buffers - Buffers to concatenate
   * @returns {ArrayBuffer} Combined buffer
   */
  _concatBuffers(...buffers) {
    const totalLength = buffers.reduce((sum, buf) => sum + buf.byteLength, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;

    for (const buffer of buffers) {
      result.set(new Uint8Array(buffer), offset);
      offset += buffer.byteLength;
    }

    return result.buffer;
  }

  /**
   * Waits for data channel to open
   * @param {RTCDataChannel} channel - Channel to wait for
   * @returns {Promise<void>}
   */
  _waitForChannelOpen(channel) {
    return new Promise((resolve, reject) => {
      if (channel.readyState === "open") {
        resolve();
        return;
      }

      const onOpen = () => {
        channel.removeEventListener("open", onOpen);
        channel.removeEventListener("error", onError);
        resolve();
      };

      const onError = (error) => {
        channel.removeEventListener("open", onOpen);
        channel.removeEventListener("error", onError);
        reject(error);
      };

      channel.addEventListener("open", onOpen);
      channel.addEventListener("error", onError);
    });
  }

  /**
   * Waits if the data channel buffer is too full
   * @param {RTCDataChannel} channel - Channel to check
   * @param {number} threshold - Buffer threshold in bytes (default 1MB)
   * @returns {Promise<void>}
   */
  _waitForBufferDrain(channel, threshold = 1024 * 1024) {
    return new Promise((resolve) => {
      const check = () => {
        if (channel.bufferedAmount < threshold) {
          resolve();
        } else {
          setTimeout(check, 10);
        }
      };
      check();
    });
  }
}

// =============================================================================
// USAGE EXAMPLE
// =============================================================================

/**
 * Example: How to use WebRTCVideoUploader with an existing peer connection
 */
async function exampleUsage() {
  // Assume you have an existing RTCPeerConnection from your WebRTC setup
  const peerConnection = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  // Create the uploader
  const uploader = new WebRTCVideoUploader(peerConnection);

  // Create the video upload channel
  const uploadChannel = uploader.createVideoUploadChannel();

  // Set up handler for the default data channel (for receiving outputs)
  peerConnection.ondatachannel = (event) => {
    if (event.channel.label !== "video_upload") {
      uploader.setupOutputHandler(event.channel);
    }
  };

  // Register callback for outputs
  uploader.onOutput((output) => {
    console.log("Received output:", output);

    // Output structure (WebRTCOutput):
    // {
    //   serialized_output_data: { ... },  // Workflow outputs
    //   video_metadata: {
    //     frame_id: number,
    //     received_at: string,
    //     pts: number,
    //     time_base: number,
    //     declared_fps: number,
    //     measured_fps: number
    //   },
    //   errors: []
    // }
  });

  // ... complete WebRTC signaling (offer/answer exchange) ...

  // Upload a video file (e.g., from file input)
  const fileInput = document.querySelector('input[type="file"]');
  fileInput.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) {
      try {
        await uploader.uploadVideoFile(file);
        console.log("Upload finished!");
      } catch (error) {
        console.error("Upload failed:", error);
      }
    }
  });
}

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = { WebRTCVideoUploader };
}
