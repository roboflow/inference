from inference.core.models.roboflow import OnnxRoboflowInferenceModel

from typing import Tuple

import numpy as np

SemanticSegmentationModelOutput = Tuple[np.ndarray]

class SemanticSegmentationBaseOnnxRoboflowInferenceModel(OnnxRoboflowInferenceModel):
    
    task_type = "semantic-segmentation"

    preprocess_means = [0.5, 0.5, 0.5]
    preprocess_stds = [0.5, 0.5, 0.5]