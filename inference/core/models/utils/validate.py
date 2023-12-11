def get_num_classes_from_model_prediction_shape(len_prediction, masks=0, keypoints=0):
    num_classes = len_prediction - 5 - masks - (keypoints * 3)
    return num_classes
