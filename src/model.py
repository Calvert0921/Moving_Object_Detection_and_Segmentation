import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# ---------------------------
# Pretrained Backbone using ResNet50
# ---------------------------
class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Backbone, self).__init__()
        # Use a pretrained ResNet50.
        # Set replace_stride_with_dilation=[False, False, True] so that layer4 uses dilated convolutions,
        # which helps maintain a higher resolution feature map.
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT, replace_stride_with_dilation=[False, False, True])
        # Initial layers: conv1, bn1, relu, and maxpool.
        self.initial = nn.Sequential(
            resnet.conv1,  # Output: (batch, 64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # Output: (batch, 64, H/4, W/4)
        )
        # Subsequent layers.
        self.layer1 = resnet.layer1  # Output: (batch, 256, H/4, W/4)
        self.layer2 = resnet.layer2  # Output: (batch, 512, H/8, W/8)
        self.layer3 = resnet.layer3  # Output: (batch, 1024, H/16, W/16)
        # With dilation, layer4 does not further downsample the feature map.
        self.layer4 = resnet.layer4  # Output: (batch, 2048, H/16, W/16)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# ---------------------------
# Segmentation Head: Decoder
# ---------------------------
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=6):
        super(SegmentationHead, self).__init__()
        # The input channels are 2048 from the ResNet50 backbone.
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Final 1x1 convolution to produce logits for each class.
        self.conv_last = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Assume input x has shape (batch, 2048, 45, 60)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # -> (batch, 2048, 90, 120)
        x = self.up1(x)  # -> (batch, 512, 90, 120)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # -> (batch, 512, 180, 240)
        x = self.up2(x)  # -> (batch, 256, 180, 240)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # -> (batch, 256, 360, 480)
        x = self.up3(x)  # -> (batch, 128, 360, 480)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # -> (batch, 128, 720, 960)
        x = self.up4(x)  # -> (batch, 64, 720, 960)
        x = self.conv_last(x)  # -> (batch, num_classes, 720, 960)
        return x

# ---------------------------
# Detection Head: Simple Anchor-based Detector
# ---------------------------
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3, num_classes=6):
        super(DetectionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Bounding box regression branch: predicts 4 values per anchor (e.g., [dx, dy, dw, dh])
        self.regression = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        # Classification branch: predicts class scores per anchor (including background)
        self.classification = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Input x shape: (batch, 2048, 45, 60)
        bbox_reg = self.regression(x)    # Output: (batch, num_anchors*4, 45, 60)
        cls_score = self.classification(x)  # Output: (batch, num_anchors*num_classes, 45, 60)
        return bbox_reg, cls_score

# ---------------------------
# Multi-Task Model: Combines Backbone with Both Heads
# ---------------------------
# class MultiTaskModel(nn.Module):
#     def __init__(self, num_classes_seg=6, num_anchors=3, num_classes_det=6):
#         super(MultiTaskModel, self).__init__()
#         self.backbone = Backbone(pretrained=True)
#         self.seg_head = SegmentationHead(in_channels=2048, num_classes=num_classes_seg)
#         self.det_head = DetectionHead(in_channels=2048, num_anchors=num_anchors, num_classes=num_classes_det)
        
#     def forward(self, x):
#         # Shared feature extraction
#         features = self.backbone(x)
#         # Segmentation branch
#         seg_out = self.seg_head(features)
#         # Detection branch
#         bbox_reg, cls_score = self.det_head(features)
#         return seg_out, bbox_reg, cls_score
    
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_seg=6, num_anchors=3, num_classes_det=6):
        super(MultiTaskModel, self).__init__()
        self.backbone = Backbone(pretrained=True)
        self.seg_head = SegmentationHead(in_channels=2048, num_classes=num_classes_seg)
        self.det_head = DetectionHead(in_channels=2048, num_anchors=num_anchors, num_classes=num_classes_det)
        
    def forward(self, x):
        # Shared feature extraction
        features = self.backbone(x)
        # Segmentation branch
        seg_out = self.seg_head(features)
        return seg_out

# ---------------------------
# Testing the Model Architecture
# ---------------------------
if __name__ == '__main__':
    # Create a dummy input image of shape (1, 3, 720, 960)
    dummy_input = torch.randn(1, 3, 720, 960)
    print(dummy_input)
    model = MultiTaskModel()
    seg_out, bbox_reg, cls_score = model(dummy_input)
    
    # Print output shapes
    print("Segmentation output shape:", seg_out.shape)  # Expected: (1, 6, 720, 960)
    print("BBox regression output shape:", bbox_reg.shape)  # Expected: (1, 12, 45, 60) [3 anchors * 4 values]
    print("Classification output shape:", cls_score.shape)  # Expected: (1, 18, 45, 60) [3 anchors * 6 classes]
