import os
import csv
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import MultiTaskModel
import matplotlib.pyplot as plt

# ----------------------------
# Read CSV Color Mapping File
# ----------------------------
def read_color_mapping(csv_path):
    """
    Reads a CSV file and returns two dictionaries:
      - A mapping from class label (integer) to color (tuple of (r, g, b)).
      - An inverse mapping from color (tuple of (r, g, b)) to class label (integer).
    CSV format:
        class,r,g,b
    """
    # Predefined mapping from name to numeric label
    predefined_mapping = {
        'Car': 1,
        'Pedestrian': 2,
        'Bicyclist': 3,
        'MotorcycleScooter': 4,
        'Truck_Bus': 5
    }
    
    mapping = {}      # numeric label -> (r, g, b)
    mapping_inv = {}  # (r, g, b) -> numeric label
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            if name in predefined_mapping:
                label = predefined_mapping[name]
                r = int(row['r'])
                g = int(row['g'])
                b = int(row['b'])
                mapping[label] = (r, g, b)
                mapping_inv[(r, g, b)] = label
    return mapping, mapping_inv

# ----------------------------
# Convert Predicted Label Image (0-5) to a Color Image
# ----------------------------
def label2color(label_img, mapping):
    """
    label_img: numpy array with pixel values in the range 0~5, where 0 represents the background.
    mapping: dict, {class: (r, g, b)}. Note that the background is fixed to (0,0,0).
    Returns a color image (numpy array) with shape (H, W, 3).
    """
    H, W = label_img.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    # Background is fixed as (0,0,0)
    for cls in range(1, 6):
        color = mapping.get(cls, (0, 0, 0))
        color_img[label_img == cls] = color
    # Background is already (0,0,0)
    return color_img

# ----------------------------
# Convert Color Label Image to Class Label Image
# (Only retains the 5 classes; all others are treated as background 0)
# ----------------------------
def color2label(color_img, mapping_inv):
    """
    color_img: numpy array with shape (H, W, 3) representing a color label image.
    mapping_inv: dict, {(r, g, b): class}
    Returns a class label image (numpy array) with pixel values 0~5, where pixels not in the mapping are treated as background (0).
    """
    H, W, _ = color_img.shape
    label_img = np.zeros((H, W), dtype=np.uint8)
    # Iterate over all pixels (a vectorized method could also be used)
    for i in range(H):
        for j in range(W):
            color = tuple(color_img[i, j])
            # If the color is in the mapping, assign the corresponding class; otherwise, treat it as background
            label_img[i, j] = mapping_inv.get(color, 0)
    return label_img

# ----------------------------
# Compare Prediction and Ground Truth Segmentation
# ----------------------------
def compare_seg(pred, gt):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pred)
    plt.title("Prediction")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(gt)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.show()

# ----------------------------
# Test Procedure
# ----------------------------
def main():
    # File path configuration
    model_checkpoint = 'weights/test_model.pth'  # Change this to the path of your trained model
    csv_path = "data/CamVid/class_dict.csv"  # CSV color mapping file path

    test_images_dir = 'data/CamVid/test'
    test_labels_dir = 'data/CamVid/test_labels'  # Folder containing the test label color images
    output_dir = 'results'  # Directory to save visualization results

    # os.makedirs(output_dir, exist_ok=True)

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

    for img_file in test_image_files:
        # Load the original image
        img_path = os.path.join(test_images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # (1,3,H,W)
        
        # Model prediction
        with torch.no_grad():
            seg_logits = model(input_tensor)
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

        # Show segmentation results
        compare_seg(pred_color, gt_color_converted)

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
