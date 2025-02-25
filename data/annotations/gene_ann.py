import json
import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

def extract_bounding_boxes(segmentation_path, csv_path, target_classes, min_area=10):
    """
    Extract bounding boxes from a segmentation image where each pixel is represented by (R,G,B).

    Parameters:
        segmentation_path (str): Path to the RGB segmentation image.
        csv_path (str): Path to the CSV file containing class RGB mappings.
                        Expected CSV format: headers 'name', 'r', 'g', 'b'.
        target_classes (set or list): Only process classes whose names are in this set.
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
            if class_name in target_classes.values():
                r = int(row['r'])
                g = int(row['g'])
                b = int(row['b'])
                for k, v in target_classes.items():
                    if v == class_name:
                        rgb_mapping[(r, g, b)] = k
    
    annotations = []
    
    # Iterate over each target RGB value and create a binary mask
    for rgb, new_class in rgb_mapping.items():
        r_val, g_val, b_val = rgb
        # Create binary mask where pixels exactly match the RGB value
        mask = ((seg[:, :, 0] == r_val) & 
                (seg[:, :, 1] == g_val) & 
                (seg[:, :, 2] == b_val)).astype(np.uint8) * 255
        
        # Find contours in the mask (external contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue  # skip small regions
            x, y, w, h = cv2.boundingRect(contour)
            annotation = {
                'bbox': [x, y, w, h],  # COCO expects [x, y, width, height]
                'category_id': new_class,
                'area': w * h,
                'iscrowd': 0
            }
            annotations.append(annotation)
    return annotations

def visualize_bbox(image_path, anns, target_classes):
    # Load the original image
    image = cv2.imread(image_path)
    # Convert from BGR (OpenCV default) to RGB for proper display with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Example list of bounding boxes, where each entry is a dict containing a 'bbox' and a 'category_id'
    # For example: [{'bbox': [x, y, w, h], 'category_id': 2}, ...]
    # Replace this with the output from your extract_bounding_boxes function.
    boxes = []
    for ann in anns:
        boxes.append({'bbox': ann['bbox'], 'category': target_classes[ann['category_id']]})

    # Draw each bounding box on the image
    for box in boxes:
        x, y, w, h = box['bbox']
        # Draw the rectangle. Color is in RGB (e.g., red here is (255, 0, 0)), thickness is 2.
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
        # Optionally, put the category id on the image
        cv2.putText(image_rgb, str(box['category']), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.show()

def create_coco_json(image_dir, segmentation_dir, csv_path, target_classes, output_json):
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define categories for the 5 target classes
    for class_id in target_classes.keys():
        coco["categories"].append({
            "id": class_id,
            "name": target_classes[class_id]
        })
    
    annotation_id = 1
    image_id = 1
    
    # Loop through each image file (assuming segmentation files have the same name)
    for file_name in os.listdir(image_dir):
        if not file_name.endswith(('.png', '.jpg')):
            continue
        name, ext = os.path.splitext(file_name)  # Splits into 'name' and '.png'
        seg_file_name = f"{name}_L{ext}"  # Creates 'name_L.png'
        image_path = os.path.join(image_dir, file_name)
        seg_path = os.path.join(segmentation_dir, seg_file_name)
        print(seg_path)
        
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
        annots = extract_bounding_boxes(seg_path, csv_path, target_classes)
        for ann in annots:
            ann["id"] = annotation_id
            ann["image_id"] = image_id
            coco["annotations"].append(ann)
            annotation_id += 1
        
        image_id += 1
    
    # Write out to a JSON file
    with open(output_json, 'w') as f:
        json.dump(coco, f)

if __name__ == "__main__":
    image_dir = 'data/CamVid/train'
    label_dir = 'data/CamVid/train_labels'
    csv_path = 'data/CamVid/class_dict.csv'
    output_json = 'data/annotations/trian_anns.json'

    # Gather all image filenames
    image_filenames = sorted(
        f for f in os.listdir(image_dir)
        if f.endswith(".png") or f.endswith(".jpg")
    )

    label_filenames = sorted(
        f for f in os.listdir(label_dir)
        if f.endswith(".png") or f.endswith(".jpg")
    )

    target_classes = {1: 'Car', 2: 'Pedestrian', 3: 'Bicyclist', 4: 'MotorcycleScooter', 5: 'Truck_Bus'}

    test_path = 'data/CamVid/train_labels/0001TP_009210_L.png'
    # anns = extract_bounding_boxes(test_path, csv_path, target_classes)
    # print(anns)
    # visualize_bbox(test_path, anns, target_classes)
    create_coco_json(image_dir, label_dir, csv_path, target_classes, output_json)
