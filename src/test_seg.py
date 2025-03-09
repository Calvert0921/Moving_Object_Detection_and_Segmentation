import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import MultiTaskModel
import matplotlib.pyplot as plt
from utils import read_color_mapping, label2color, color2label
from tqdm import tqdm

# ----------------------------
# Save the comparison of Prediction and Ground Truth Segmentation
# ----------------------------
def save_comparison(pred, gt, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(pred)
    axs[0].set_title("Prediction")
    axs[0].axis("off")
    axs[1].imshow(gt)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Test Procedure
# ----------------------------
def main():
    # File path configuration
    model_checkpoint = '../weights/segmentation_model.pth'  # Change this to the path of your trained model
    csv_path = "../data/CamVid/class_dict.csv"  # CSV color mapping file path

    test_images_dir = '../data/CamVid/test'
    test_labels_dir = '../data/CamVid/test_labels'  # Folder containing the test label color images
    output_dir = '../results'  # Directory to save visualization results
    os.makedirs(output_dir, exist_ok=True)

    # Load color mapping
    color_mapping, color_mapping_inv = read_color_mapping(csv_path)
    
    # Define image transformation (must be consistent with training)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load test data (here we simply iterate over the folder)
    test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Initialize accumulators for IoU computation for classes 1-5 (ignore background)
    # We use 6 slots so that index corresponds to class number (index 0 for background)
    class_intersections = np.zeros(6, dtype=np.float32)
    class_unions = np.zeros(6, dtype=np.float32)

    # Loop over test images with tqdm progress bar
    for img_file in tqdm(test_image_files, desc="Processing images"):
        # Load the original image
        img_path = os.path.join(test_images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # (1,3,H,W)
        
        # Model prediction
        with torch.no_grad():
            seg_logits, _, _ = model(input_tensor)
            # Get the predicted label image (shape: (1, H, W))
            pred_labels = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # Value range: 0~5

        # Convert predicted label image to a color image
        pred_color = label2color(pred_labels, color_mapping)
        
        # Load the corresponding test label image (color)
        gt_path = os.path.join(test_labels_dir, os.path.splitext(img_file)[0] + '_L.png')
        gt_color = np.array(Image.open(gt_path).convert('RGB'))
        # Convert ground truth to a class label image; only retain the 5 classes, treating others as background
        gt_labels = color2label(gt_color, color_mapping_inv)
        # Optionally, convert the ground truth back to a color image for comparison
        gt_color_converted = label2color(gt_labels, color_mapping)

        # Accumulate intersections and unions for each class (only classes 1-5)
        for cls in range(1, 6):
            pred_cls = (pred_labels == cls)
            gt_cls = (gt_labels == cls)
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            class_intersections[cls] += intersection
            class_unions[cls] += union

        # Save comparison results
        save_file = os.path.join(output_dir, os.path.splitext(img_file)[0] + '_comparison.png')
        save_comparison(pred_color, gt_color_converted, save_file)

    # Compute and print per-class IoU and average IoU (only for classes 1-5)
    print("Per-class IoU (excluding background):")
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
        print(f"Average IoU (classes 1-5): {average_iou:.4f}")
    else:
        print("No valid classes found for IoU calculation.")

if __name__ == '__main__':
    main()
