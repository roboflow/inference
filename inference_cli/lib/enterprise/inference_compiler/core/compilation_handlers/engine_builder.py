import logging
import os
from typing import Any, Dict, Literal, Optional, Tuple

import tensorrt as trt

from inference_cli.lib.enterprise.inference_compiler.core.compilation_handlers.timing_cache_manager import (
    TimingCacheManager,
)
from inference_cli.lib.enterprise.inference_compiler.errors import (
    InvalidNetworkInputsError,
    NetworkParsingError,
    QuantizationNotSupportedError,
    TRTModelCompilationError,
)
from inference_models.logger import LOGGER
from inference_models.models.common.trt import InferenceTRTLogger

LOGGER.setLevel(logging.INFO)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(
        self,
        workspace: int,
    ):
        self.trt_logger = InferenceTRTLogger()
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )
        self.network: Optional[trt.tensorrt.INetworkDefinition] = None
        self.parser: Optional[trt.OnnxParser] = None
        self.cache_manager: Optional[TimingCacheManager] = None

    def set_timing_cache_manager(self, cache_manager: TimingCacheManager) -> None:
        self.cache_manager = cache_manager

    def create_network(self, onnx_path: str) -> None:
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        self.network = self.builder.create_network(0)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                LOGGER.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    LOGGER.error(self.parser.get_error(error))
                raise NetworkParsingError("Could not parse ONNX file")

        network_inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]
        network_outputs = [
            self.network.get_output(i) for i in range(self.network.num_outputs)
        ]
        LOGGER.info("Network Description")
        for network_input in network_inputs:
            LOGGER.info(
                "Input '{}' with shape {} and dtype {}".format(
                    network_input.name, network_input.shape, network_input.dtype
                )
            )
        for network_output in network_outputs:
            LOGGER.info(
                "Output '{}' with shape {} and dtype {}".format(
                    network_output.name, network_output.shape, network_output.dtype
                )
            )

    def get_static_batch_size_of_input(self) -> int:
        network_input = self._get_image_input()
        try:
            return int(network_input.shape[0])
        except ValueError as error:
            raise InvalidNetworkInputsError(
                f"Expected the input to have static batch size, detected shape: {network_input.shape}"
            ) from error

    def create_engine(
        self,
        engine_path: str,
        precision: Literal["fp32", "fp16"],
        input_size: Tuple[int, int],
        dynamic_batch_sizes: Optional[Tuple[int, int, int]] = None,
        trt_version_compatible: bool = False,
        same_compute_compatibility: bool = False,
    ) -> None:
        if self.cache_manager:
            cache_bytes = self.cache_manager.get_cache_for_features()
            cache = self.config.create_timing_cache(cache_bytes)
            self.config.set_timing_cache(cache, ignore_mismatch=False)
        engine_path = os.path.abspath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        LOGGER.info("Building {} Engine in {}".format(precision, engine_path))
        network_input = self._get_image_input()
        input_name = network_input.name
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                raise QuantizationNotSupportedError("FP16 quantization not supported")
            self.config.set_flag(trt.BuilderFlag.FP16)
        if trt_version_compatible:
            self.config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        if same_compute_compatibility:
            self.config.hardware_compatibility_level = (
                trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
            )
        profile = self.builder.create_optimization_profile()
        if dynamic_batch_sizes:
            bs_min, bs_opt, bs_max = dynamic_batch_sizes
            h, w = input_size
            profile.set_shape(
                input_name, (bs_min, 3, h, w), (bs_opt, 3, h, w), (bs_max, 3, h, w)
            )
        self.config.add_optimization_profile(profile)
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            raise TRTModelCompilationError("Failed to create TRT engine")
        with open(engine_path, "wb") as f:
            LOGGER.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)
        if self.cache_manager:
            cache = self.config.get_timing_cache()
            self.cache_manager.save_cache_for_features(cache=cache.serialize())

    def _get_image_input(self) -> trt.ITensor:
        if self.network is None:
            raise TRTModelCompilationError(
                "Attempted to get network input before parsing the model"
            )
        network_inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]
        if len(network_inputs) != 1:
            raise InvalidNetworkInputsError("Detected network with multiple inputs")
        return network_inputs[0]
