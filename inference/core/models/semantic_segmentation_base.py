from inference.core.models.instance_segmentation_base import InstanceSegmentationBaseOnnxRoboflowInferenceModel

class SemanticSegmentationBaseOnnxRoboflowInferenceModel(InstanceSegmentationBaseOnnxRoboflowInferenceModel):
    
    task_type = "semantic-segmentation"