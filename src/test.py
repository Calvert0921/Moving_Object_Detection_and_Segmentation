import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm
import torch.nn.functional as F

from model import MultiTaskModel
from utils import read_color_mapping, label2color, color2label
from evaluate import nms, generate_anchors, compute_map, decode_boxes
from dataloader import get_dataloader

# -------------------------------------------
# Helper: Get detection predictions for one image
# -------------------------------------------
def get_detection_predictions(bbox_pred, cls_pred, anchors, score_threshold=0.05, iou_threshold=0.5, device='cpu'):
    """
    For a single image, decode detection predictions, apply score threshold and NMS.
    Returns:
        detections: list of detections, each a tuple (box, score, cls).
    """
    detections = []
    scores = F.softmax(cls_pred, dim=-1)  # shape (N, num_classes)
    
    # Process classes 1-5 (ignore background)
    for cls in range(1, 6):
        cls_scores = scores[:, cls]
        inds = (cls_scores > score_threshold).nonzero(as_tuple=True)[0]
        if inds.numel() == 0:
            continue
        cls_boxes = decode_boxes(anchors.to(device), bbox_pred)[inds]
        cls_scores_selected = cls_scores[inds]
        keep = nms(cls_boxes, cls_scores_selected, iou_threshold=iou_threshold)
        final_boxes = cls_boxes[keep].cpu()
        final_scores = cls_scores_selected[keep].cpu()
        for box, score in zip(final_boxes.tolist(), final_scores.tolist()):
            detections.append((box, score, cls))
    
    return detections





# -------------------------------------------
# Save the comparison image with segmentation and detection boxes.
# -------------------------------------------
def save_comparison(pred_seg, gt_seg, detection_detections, save_path):
    """
    Saves a comparison image with the left panel showing the segmentation prediction
    (with detection boxes overlaid) and the right panel showing the ground truth segmentation.
    
    Args:
        pred_seg (ndarray): Predicted segmentation color image.
        gt_seg (ndarray): Ground truth segmentation color image.
        detection_detections (list): List of detection tuples (box, score, cls).
        save_path (str): Path to save the comparison image.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(pred_seg)
    axs[0].set_title("Segmentation Prediction\n(with Detection Boxes)")
    axs[0].axis("off")
    
    # Overlay detection boxes on the segmentation prediction.
    for box, score, cls in detection_detections:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none')
        axs[0].add_patch(rect)
        axs[0].text(xmin, ymin - 5, f"{cls}:{score:.2f}", color='g', fontsize=9, backgroundcolor='black')
    
    axs[1].imshow(gt_seg)
    axs[1].set_title("Ground Truth Segmentation")
    axs[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# -------------------------------------------
# Test Procedure
# -------------------------------------------
def main():
    # File path configuration.
    model_checkpoint = '../weights/final_model.pth'  # Path to your trained model.
    csv_path = "../data/CamVid/class_dict.csv"              # CSV color mapping file path.

    test_images_dir = '../data/CamVid/test'
    test_labels_dir = '../data/CamVid/test_labels'            # Folder with test label color images.
    output_dir = '../results'                                 # Directory to save visualization results.
    os.makedirs(output_dir, exist_ok=True)

    # For detection evaluation, assume a COCO-style JSON exists and a corresponding masks dir.
    test_annotations_file = '../data/annotations/test_anns.json'
    test_masks_dir = '../data/annotations/test_masks'

    # Load color mapping for segmentation visualization.
    color_mapping, color_mapping_inv = read_color_mapping(csv_path)
    
    # Define image transformation (must be consistent with training).
    transform = transforms.Compose([transforms.ToTensor()])
    
    # List test image files.
    test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    # Load model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate anchors for detection (assume feature size, stride, scales as in training).
    anchors = generate_anchors((45, 60), 16, [32, 64, 128])  # Shape: (N, 4) in [xmin, ymin, xmax, ymax]
    anchors = anchors.to(device)

    # Initialize accumulators for segmentation IoU computation for classes 1-5 (ignore background).
    class_intersections = np.zeros(6, dtype=np.float32)
    class_unions = np.zeros(6, dtype=np.float32)

    # Process each test image.
    for img_file in tqdm(test_image_files, desc="Processing test images"):
        # Load original image.
        img_path = os.path.join(test_images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, 3, H, W)
        
        with torch.no_grad():
            seg_logits, bbox_pred, cls_pred = model(input_tensor)
            # Segmentation prediction.
            pred_labels = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # Values: 0~5

        # Convert predicted label image to a color image.
        pred_color = label2color(pred_labels, color_mapping)
        
        # Load ground truth segmentation (color image) and convert to labels.
        gt_path = os.path.join(test_labels_dir, os.path.splitext(img_file)[0] + '_L.png')
        gt_color = np.array(Image.open(gt_path).convert('RGB'))
        gt_labels = color2label(gt_color, color_mapping_inv)
        gt_color_converted = label2color(gt_labels, color_mapping)
        
        # Accumulate segmentation intersections/unions per class.
        for cls in range(1, 6):
            pred_cls = (pred_labels == cls)
            gt_cls = (gt_labels == cls)
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            class_intersections[cls] += intersection
            class_unions[cls] += union

        # ----------------------------
        # Detection Predictions for the current image.
        # ----------------------------
        # Reshape detection outputs.
        # bbox_pred: (batch, num_anchors*4, 45, 60) -> reshape to (N, 4)
        bbox_pred_img = bbox_pred.view(-1, 4)
        # cls_pred: (batch, num_anchors*num_classes, 45, 60) -> reshape to (N, num_classes)
        cls_pred_img = cls_pred.view(-1, 6)
        
        # Use the provided decode_boxes function inside get_detection_predictions.
        detection_detections = get_detection_predictions(bbox_pred_img, cls_pred_img, anchors,
                                                         score_threshold=0.7, iou_threshold=0.5,
                                                         device=device)
        
        # Save comparison image with segmentation and detection boxes.
        save_file = os.path.join(output_dir, os.path.splitext(img_file)[0] + '_comparison.png')
        save_comparison(pred_color, gt_color_converted, detection_detections, save_file)
    
    # Compute and print segmentation IoU per class and average IoU.
    print("Segmentation Per-class IoU (excluding background):")
    iou_list = []
    for cls in range(1, 6):
        if class_unions[cls] > 0:
            iou = class_intersections[cls] / class_unions[cls]
            print(f"Class {cls}: IoU = {iou:.4f}")
            iou_list.append(iou)
        else:
            print(f"Class {cls}: IoU = N/A (no samples)")
    if iou_list:
        average_iou = sum(iou_list) / len(iou_list)
        print(f"Average Segmentation IoU (classes 1-5): {average_iou:.4f}")
    else:
        print("No valid classes found for segmentation IoU calculation.")

    # ----------------------------
    # Detection mAP Computation
    # ----------------------------
    # Create a dataloader for detection evaluation.
    test_loader = get_dataloader(test_images_dir, test_masks_dir, test_annotations_file,
                                 batch_size=4, num_workers=4, shuffle=False)
    detection_mAP = compute_map(model, test_loader, device, anchors,
                                iou_threshold=0.5, score_threshold=0.05)
    print(f"Detection mAP (classes 1-5): {detection_mAP:.4f}")

if __name__ == '__main__':
    main()
