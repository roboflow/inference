import os.path
from typing import Literal, Optional, List
import os
import sys
import logging
import argparse

import cv2
import tensorrt as trt
import torch
from torchvision.transforms import functional

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
logger = logging.getLogger("EngineBuilder")


def load_images_paths(images_list_file: str) -> List[str]:
    with open(images_list_file) as f:
        return [l.strip() for l in f.readlines() if len(l.strip())]


class EngineCalibrator(trt.IInt8MinMaxCalibrator):
    """
    Implements the INT8 MinMax Calibrator.
    """

    def __init__(
        self,
        cache_file: str,
        images_list_file: str,
    ):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.last_yielded = 0
        self.images = load_images_paths(images_list_file)
        self.loaded_images = []

    def get_batch_size(self):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if self.last_yielded >= len(self.images):
            return None
        batch_tensors = []
        for _ in range(512):
            logger.info(f"Calibrating image {self.last_yielded} / {len(self.images)}")
            image = torch.from_numpy(cv2.imread(self.images[self.last_yielded]))
            image_rgb = image[..., [2, 1, 0]]
            rgb_chw = image_rgb.permute(2, 0, 1) / 255.0
            _, h, w = rgb_chw.shape
            target_h, target_w = (64, 64)
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image_tensor = functional.resize(rgb_chw, [new_h, new_w], interpolation=functional.InterpolationMode.BILINEAR)
            pad_top = max((target_h - new_h) // 2, 0)
            pad_bottom = max(target_h - new_h - pad_top, 0)
            pad_left = max((target_w - new_w) // 2, 0)
            pad_right = max(target_w - new_w - pad_left, 0)
            image_tensor = functional.pad(image_tensor, [pad_left, pad_top, pad_right, pad_bottom], fill=0.5)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            image_tensor = image_tensor.contiguous().to("cuda")
            batch_tensors.append(image_tensor)
            self.last_yielded += 1
            if self.last_yielded >= len(self.images):
                return None
        result = torch.cat(batch_tensors, dim=0)
        self.loaded_images.append(result)
        result = [result.data_ptr()]
        torch.cuda.synchronize()
        return result

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            logger.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose: bool = True, workspace: int = 8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        self.network = self.builder.create_network(0)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                logger.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    logger.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        logger.info("Network Description")
        for input in inputs:
            logger.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            logger.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )

    def create_engine(
        self,
        engine_path,
        images_list_file,
        precision,
        calib_cache=None,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16', 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        logger.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision in ["fp16", "int8"]:
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)
        if precision in ["int8"]:
            if not self.builder.platform_has_fast_int8:
                logger.warning("INT8 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.INT8)
            self.config.int8_calibrator = EngineCalibrator(cache_file=calib_cache, images_list_file=images_list_file)

        profile = self.builder.create_optimization_profile()
        profile.set_shape("images", (1, 3, 64, 64), (512, 3, 64, 64), (1024, 3, 64, 64))
        self.config.add_optimization_profile(profile)
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            logger.error("Failed to create engine")
            sys.exit(1)

        with open(engine_path, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)


def main(
    onnx_path: str,
    engine_path: str,
    precision: Literal["fp32", "fp16", "int8"] = "fp32",
    workspace: int = 8,
    calibration_images: Optional[str] = None,
    calibration_cache: str = "./calibration.cache"
) -> None:
    if precision == "int8" and calibration_images is None or not os.path.exists(calibration_images):
        raise RuntimeError("Could not compile engine in int8 mode without calibration images.")
    builder = EngineBuilder(verbose=True, workspace=workspace)
    builder.create_network(onnx_path=onnx_path)
    builder.create_engine(
        engine_path=engine_path,
        images_list_file=calibration_images,
        precision=precision,
        calib_cache=calibration_cache,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--engine-path", type=str, required=True)
    parser.add_argument("--precision", type=str, required=False, default="fp32")
    parser.add_argument("--workspace", type=int, required=False, default=8)
    parser.add_argument("--calibration-images", type=str, required=False)
    parser.add_argument("--calibration-cache", type=str, required=False, default="./calibration.cache")
    args = parser.parse_args()
    main(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        precision=args.precision,
        workspace=args.workspace,
        calibration_images=args.calibration_images,
        calibration_cache=args.calibration_cache,
    )





