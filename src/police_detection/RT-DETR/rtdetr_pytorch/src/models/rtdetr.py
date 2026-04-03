
import torch
import torch.nn as nn
import torchvision.models as models
from src.core.yaml_utils import register  # Ensure register is used

@register
class RTDETR_R50VD(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        # Backbone (ResNet-50 without FC head)
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool & fc

        # Conv head to predict logits and boxes
        self.conv = nn.Conv2d(2048, 256, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Detection head
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)  # Class logits
        self.box_head = nn.Conv2d(256, 4, kernel_size=1)  # Bounding box (cx, cy, w, h)

    def forward(self, samples, targets=None):
        if isinstance(samples, dict):
            images = samples["image"]
        else:
            images = samples

        features = self.backbone(images)  # [B, 2048, H/32, W/32]
        x = self.relu(self.norm(self.conv(features)))  # [B, 256, H/32, W/32]

        logits = self.cls_head(x)  # [B, num_classes, H/32, W/32]
        boxes = self.box_head(x)   # [B, 4, H/32, W/32]

        # Flatten outputs
        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)  # [B, HW, num_classes]
        boxes = boxes.permute(0, 2, 3, 1).reshape(B, -1, 4)  # [B, HW, 4]

        out = {
            "pred_logits": logits,
            "pred_boxes": boxes
        }

        return out

# class RTDETR_R50VD(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
#         self.num_classes = num_classes
#         # TODO: define your backbone, encoder, decoder, etc.
#         self.backbone = nn.Identity()  # placeholder
#         self.head = nn.Linear(640, num_classes)  # adjust based on actual output

#     # def forward(self, samples):
#     # x = self.backbone(samples["image"])  # or just samples if already preprocessed
#     # logits, boxes = self.head(x)

#     # return {
#     #     "pred_logits": logits,
#     #     "pred_boxes": boxes
#     # }

#     def forward(self, samples, targets=None):
#         # Step 1: Feature extraction
#         ffeatures = self.backbone(samples)  # or samples["image"] if samples is a dict

#         outputs = self.head(ffeatures)

#         print("HEAD OUTPUT TYPE:", type(outputs))
#         if isinstance(outputs, tuple):
#             print("HEAD OUTPUT LENGTH:", len(outputs))
#             for i, out in enumerate(outputs):
#                 print(f"Output {i} shape:", getattr(out, 'shape', type(out)))
#         else:
#             print("HEAD OUTPUT SHAPE:", getattr(outputs, 'shape', type(outputs)))

#         # # Optional: do something during training
#         # if self.training and targets is not None:
#         #     # possibly preprocess or validate targets
#         #     pass

#         # return out_dict

    
    

