import logging
import os
import time
from typing import Literal, Optional, Tuple

import tensorrt as trt
from inference_exp.logger import LOGGER
from inference_exp.models.common.trt import InferenceTRTLogger

LOGGER.setLevel(logging.INFO)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, workspace: int = 8):
        self.trt_logger = InferenceTRTLogger()
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.network = None
        self.parser = None

    def create_network(self, onnx_path: str) -> None:
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        LOGGER.info("Starting ONNX parsing from: {}".format(onnx_path))
        self.network = self.builder.create_network(0)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            LOGGER.info("Parsing ONNX model graph...")
            if not self.parser.parse(f.read()):
                LOGGER.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    LOGGER.error(self.parser.get_error(error))
                raise RuntimeError("Could not parse ONNX file")
            LOGGER.info("ONNX parsing completed successfully")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        LOGGER.info("Network Description")
        for input in inputs:
            LOGGER.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            LOGGER.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )

    def create_engine(
        self,
        engine_path: str,
        precision: Literal["fp32", "fp16", "int8"],
        input_name: str,
        input_size: Tuple[int, int],
        dynamic_batch_sizes: Optional[Tuple[int, int, int]] = None,
        trt_version_compatible: bool = False,
        same_compute_compatibility: bool = False,
    ) -> None:
        engine_path = os.path.abspath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)

        LOGGER.info("=" * 60)
        LOGGER.info("Starting TensorRT Engine Compilation")
        LOGGER.info("=" * 60)
        LOGGER.info("Output path: {}".format(engine_path))
        LOGGER.info("Precision: {}".format(precision.upper()))
        LOGGER.info("Input size: {}x{}".format(input_size[0], input_size[1]))
        if dynamic_batch_sizes:
            LOGGER.info("Dynamic batch sizes: min={}, opt={}, max={}".format(
                dynamic_batch_sizes[0], dynamic_batch_sizes[1], dynamic_batch_sizes[2]
            ))
        else:
            LOGGER.info("Using static batch size")
        LOGGER.info("TRT version compatible: {}".format(trt_version_compatible))
        LOGGER.info("Same compute compatibility: {}".format(same_compute_compatibility))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        if len(inputs) != 1:
            raise ValueError("Detected network with multiple inputs")

        LOGGER.info("Configuring builder flags...")
        if precision in ["fp16", "int8"]:
            if not self.builder.platform_has_fast_fp16:
                LOGGER.warning("FP16 is not supported natively on this platform/device")
            else:
                LOGGER.info("FP16 is supported on this platform")
            self.config.set_flag(trt.BuilderFlag.FP16)
        if precision in ["int8"]:
            if not self.builder.platform_has_fast_int8:
                LOGGER.warning("INT8 is not supported natively on this platform/device")
            else:
                LOGGER.info("INT8 is supported on this platform")
            self.config.set_flag(trt.BuilderFlag.INT8)
        if trt_version_compatible:
            LOGGER.info("Enabling TRT version compatibility flag")
            self.config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        if same_compute_compatibility:
            LOGGER.info("Enabling same compute capability compatibility")
            self.config.hardware_compatibility_level = (
                trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
            )

        LOGGER.info("Creating optimization profile...")
        profile = self.builder.create_optimization_profile()
        if dynamic_batch_sizes:
            bs_min, bs_opt, bs_max = dynamic_batch_sizes
            h, w = input_size
            profile.set_shape(
                input_name, (bs_min, 3, h, w), (bs_opt, 3, h, w), (bs_max, 3, h, w)
            )
            LOGGER.info("Optimization profile configured with dynamic batch sizes")
        self.config.add_optimization_profile(profile)

        LOGGER.info("Building TensorRT engine - this may take several minutes...")
        start_time = time.time()
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        build_time = time.time() - start_time

        if engine_bytes is None:
            raise ValueError("Failed to create image")

        LOGGER.info("TensorRT engine built successfully in {:.2f} seconds".format(build_time))
        LOGGER.info("Engine size: {:.2f} MB".format(len(engine_bytes) / (1024 * 1024)))

        with open(engine_path, "wb") as f:
            LOGGER.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)

        LOGGER.info("=" * 60)
        LOGGER.info("TensorRT Compilation Complete")
        LOGGER.info("=" * 60)
