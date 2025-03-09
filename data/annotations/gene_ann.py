import json
import os
import cv2
import numpy as np
import csv

def extract_bounding_boxes(segmentation_path, csv_path, target_classes, mask_save_dir, min_area=10):
    """
    Extract bounding boxes from a segmentation image where each pixel is represented by (R,G,B).

    Parameters:
        segmentation_path (str): Path to the RGB segmentation image.
        csv_path (str): Path to the CSV file containing class RGB mappings.
                        Expected CSV format: headers 'name', 'r', 'g', 'b'.
        target_classes (dict): Mapping of class IDs to class names.
        mask_save_dir (str): Directory where the combined mask image will be saved.
        min_area (int): Minimum area for a contour to be considered (to filter out noise).

    Returns:
        annotations (list): A list of dictionaries with keys:
            'bbox'         : [x, y, width, height]
            'category_id'  : The class ID from the CSV mapping.
            'area'         : Area of the bounding box.
            'iscrowd'      : Set to 0 (standard for non-crowd annotations).
    """
    # Load the segmentation image in color (OpenCV loads in BGR by default)
    seg = cv2.imread(segmentation_path, cv2.IMREAD_COLOR)
    if seg is None:
        raise ValueError(f"Unable to load image at {segmentation_path}")
    # Convert image from BGR to RGB
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    
    # Build a mapping from RGB tuple to class id for target classes
    rgb_mapping = {}  # keys: (r, g, b), value: class_id
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_name = str(row['name'])
            # Check if this class name is one of our target classes
            if class_name in target_classes.values():
                r = int(row['r'])
                g = int(row['g'])
                b = int(row['b'])
                # Find the corresponding key for this class name
                for key, val in target_classes.items():
                    if val == class_name:
                        rgb_mapping[(r, g, b)] = key
    
    annotations = []
    
    # Create a combined mask (assumes image size 720x960; adjust as needed)
    combined_mask = np.zeros([seg.shape[0], seg.shape[1]])
    for rgb, class_id in rgb_mapping.items():
        r_val, g_val, b_val = rgb
        # Create binary mask where pixels exactly match the RGB value
        matched_mask = ((seg[:, :, 0] == r_val) & 
                        (seg[:, :, 1] == g_val) & 
                        (seg[:, :, 2] == b_val))
        
        # Update the combined mask with the class id
        combined_mask[matched_mask] = class_id

        mask = matched_mask.astype(np.uint8) * 255
        
        # Find contours in the mask (external contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue  # skip small regions
            x, y, w, h = cv2.boundingRect(contour)
            annotation = {
                'bbox': [x, y, w, h],  # COCO expects [x, y, width, height]
                'category_id': class_id,
                'area': w * h,
                'iscrowd': 0
            }
            annotations.append(annotation)

    # Save the combined mask in the specified directory
    os.makedirs(mask_save_dir, exist_ok=True)
    filename = os.path.basename(segmentation_path)
    name_without_ext = os.path.splitext(filename)[0]
    base_name = name_without_ext.split("_L")[0]
    new_filename = base_name + "_M.png"
    save_path = os.path.join(mask_save_dir, new_filename)
    cv2.imwrite(save_path, combined_mask.astype(np.uint8))

    return annotations

def create_coco_json(image_dir, segmentation_dir, csv_path, target_classes, output_json, mask_save_dir):
    """
    Create a COCO-style JSON file for a given dataset split.

    Parameters:
        image_dir (str): Directory containing original images.
        segmentation_dir (str): Directory containing segmentation labels.
        csv_path (str): CSV file mapping class names to RGB values.
        target_classes (dict): Mapping of class IDs to class names.
        output_json (str): Output JSON file name.
        mask_save_dir (str): Directory where combined mask images will be saved.
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define categories for the target classes
    for class_id in target_classes.keys():
        coco["categories"].append({
            "id": class_id,
            "name": target_classes[class_id]
        })
    
    annotation_id = 1
    image_id = 1
    
    # Loop through each image file (assuming corresponding segmentation files exist)
    for file_name in os.listdir(image_dir):
        if not file_name.endswith(('.png', '.jpg')):
            continue
        name, ext = os.path.splitext(file_name)
        seg_file_name = f"{name}_L{ext}"  # Assumes segmentation files are named like 'image_L.png'
        image_path = os.path.join(image_dir, file_name)
        seg_path = os.path.join(segmentation_dir, seg_file_name)
        
        print(f"Processing segmentation file: {seg_path}")
        
        # Read image to get width and height
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })
        
        # Extract bounding boxes from the segmentation mask
        annots = extract_bounding_boxes(seg_path, csv_path, target_classes, mask_save_dir)
        for ann in annots:
            ann["id"] = annotation_id
            ann["image_id"] = image_id
            coco["annotations"].append(ann)
            annotation_id += 1
        
        image_id += 1
    
    # Write out the annotations to a JSON file
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"Saved COCO annotations to {output_json}")

if __name__ == "__main__":
    # Define your CSV file and target classes mapping
    csv_path = '../CamVid/class_dict.csv'
    target_classes = {
        1: 'Car',
        2: 'Pedestrian',
        3: 'Bicyclist',
        4: 'MotorcycleScooter',
        5: 'Truck_Bus'
    }
    
    # Define the dataset splits and their directories
    splits = {
        "train": {
            "image_dir": "../CamVid/train",
            "segmentation_dir": "../CamVid/train_labels",
            "mask_save_dir": "../CamVid/train_masks",
            "output_json": "train_anns.json"
        },
        "val": {
            "image_dir": "../CamVid/val",
            "segmentation_dir": "../CamVid/val_labels",
            "mask_save_dir": "../CamVid/val_masks",
            "output_json": "val_anns.json"
        },
        "test": {
            "image_dir": "../CamVid/test",
            "segmentation_dir": "../CamVid/test_labels",
            "mask_save_dir": "../CamVid/test_masks",
            "output_json": "test_anns.json"
        }
    }
    
    # Loop over each split and generate the COCO JSON annotations
    for split, paths in splits.items():
        print(f"\n=== Generating annotations for {split} split ===")
        create_coco_json(
            image_dir=paths["image_dir"],
            segmentation_dir=paths["segmentation_dir"],
            csv_path=csv_path,
            target_classes=target_classes,
            output_json=paths["output_json"],
            mask_save_dir=paths["mask_save_dir"]
        )
