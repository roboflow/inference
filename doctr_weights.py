import torch
from doctr.models import db_resnet50, crnn_vgg16_bn

# Get v2 weights (doctr 1.0+ must be installed)
det_model = db_resnet50(pretrained=True)
torch.save(det_model.state_dict(), "db_resnet50_v2.pt")

rec_model = crnn_vgg16_bn(pretrained=True)
torch.save(rec_model.state_dict(), "crnn_vgg16_bn_v2.pt")
