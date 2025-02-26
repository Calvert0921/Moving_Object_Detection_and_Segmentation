import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone: Custom CNN Encoder
# ---------------------------
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # Each block downsamples the input by a factor of 2.
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (720,960) -> (360,480)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (360,480) -> (180,240)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (180,240) -> (90,120)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (90,120) -> (45,60)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)  # Output: (batch, 64, 360, 480)
        x = self.conv2(x)  # Output: (batch, 128, 180, 240)
        x = self.conv3(x)  # Output: (batch, 256, 90, 120)
        x = self.conv4(x)  # Output: (batch, 512, 45, 60)
        return x

# ---------------------------
# Segmentation Head: Decoder
# ---------------------------
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=6):
        super(SegmentationHead, self).__init__()
        # Four upsampling stages to recover the original resolution
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Final 1x1 convolution to produce segmentation logits for 6 classes.
        self.conv_last = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x: (batch, 512, 45, 60)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (batch, 512, 90, 120)
        x = self.up1(x)  # (batch, 256, 90, 120)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (batch, 256, 180, 240)
        x = self.up2(x)  # (batch, 128, 180, 240)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (batch, 128, 360, 480)
        x = self.up3(x)  # (batch, 64, 360, 480)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (batch, 64, 720, 960)
        x = self.up4(x)  # (batch, 32, 720, 960)
        x = self.conv_last(x)  # (batch, 6, 720, 960)
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
        # Classification branch: predicts class scores per anchor (including background as one of the classes)
        self.classification = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: (batch, 512, 45, 60)
        bbox_reg = self.regression(x)    # (batch, num_anchors*4, 45, 60)
        cls_score = self.classification(x)  # (batch, num_anchors*num_classes, 45, 60)
        return bbox_reg, cls_score

# ---------------------------
# Multi-Task Model: Combines Backbone with Both Heads
# ---------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_seg=6, num_anchors=3, num_classes_det=6):
        super(MultiTaskModel, self).__init__()
        self.backbone = Backbone()
        self.seg_head = SegmentationHead(in_channels=512, num_classes=num_classes_seg)
        self.det_head = DetectionHead(in_channels=512, num_anchors=num_anchors, num_classes=num_classes_det)
        
    def forward(self, x):
        # Shared feature extraction
        features = self.backbone(x)
        # Semantic segmentation output (logits for 6 classes)
        seg_out = self.seg_head(features)
        # Detection outputs: bounding box regression and class scores
        bbox_reg, cls_score = self.det_head(features)
        return seg_out, bbox_reg, cls_score

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
