"""Image blocks: crops, slices, resizing, preprocessing, invert, mosaic, brightness,
grayscale, blur, threshold, morphology and contrast."""

from roboflow_workflows.v2.blocks.image.primitives.crop.v1 import CropBlock
from roboflow_workflows.v2.blocks.image.primitives.has_brightness.v1 import HasBrightnessBlock
from roboflow_workflows.v2.blocks.image.primitives.invert.v1 import InvertBlock
from roboflow_workflows.v2.blocks.image.primitives.mosaic.v1 import MosaicBlock
from roboflow_workflows.v2.blocks.image.primitives.resize.v1 import ResizeBlock
from roboflow_workflows.v2.blocks.image.primitives.static_crop.v1 import StaticCropBlock
from roboflow_workflows.v2.blocks.image.image_blur.v1 import ImageBlurBlock
from roboflow_workflows.v2.blocks.image.contrast_enhancement.v1 import ContrastEnhancementBlock
from roboflow_workflows.v2.blocks.image.contrast_equalization.v1 import ContrastEqualizationBlock
from roboflow_workflows.v2.blocks.image.image_copy.v1 import ImageCopyBlock
from roboflow_workflows.v2.blocks.image.dynamic_crop.v1 import DynamicCropBlock
from roboflow_workflows.v2.blocks.image.convert_grayscale.v1 import ConvertGrayscaleBlock
from roboflow_workflows.v2.blocks.image.mask_edge_snap.v1 import MaskEdgeSnapBlock
from roboflow_workflows.v2.blocks.image.morphological_transformation.v2 import MorphologicalTransformationBlock
from roboflow_workflows.v2.blocks.image.image_preprocessing.v1 import ImagePreprocessingBlock
from roboflow_workflows.v2.blocks.image.image_slicer.v2 import ImageSlicerBlock
from roboflow_workflows.v2.blocks.image.absolute_static_crop.v1 import AbsoluteStaticCropBlock
from roboflow_workflows.v2.blocks.image.relative_statoic_crop.v1 import RelativeStaticCropBlock
from roboflow_workflows.v2.blocks.image.threshold.v1 import ImageThresholdBlock

from roboflow_workflows.v2.blocks.image.auto_rotate_on_edges.v1 import AutoRotateOnEdgesBlock
from roboflow_workflows.v2.blocks.image.camera_focus.v2 import CameraFocusBlock
from roboflow_workflows.v2.blocks.image.dominant_color.v1 import DominantColorBlock
from roboflow_workflows.v2.blocks.image.stitch_images.v1 import StitchImagesBlock

__all__ = [
    "AbsoluteStaticCropBlock",
    "ContrastEnhancementBlock",
    "ContrastEqualizationBlock",
    "ConvertGrayscaleBlock",
    "CropBlock",
    "DynamicCropBlock",
    "HasBrightnessBlock",
    "ImageBlurBlock",
    "ImageCopyBlock",
    "ImagePreprocessingBlock",
    "ImageSlicerBlock",
    "ImageThresholdBlock",
    "InvertBlock",
    "MorphologicalTransformationBlock",
    "MosaicBlock",
    "RelativeStaticCropBlock",
    "ResizeBlock",
    "StaticCropBlock",
    "MaskEdgeSnapBlock",
    "CameraFocusBlock",
    "StitchImagesBlock",
    "AutoRotateOnEdgesBlock",
    "DominantColorBlock",
]
