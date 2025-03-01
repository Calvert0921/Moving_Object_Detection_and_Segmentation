import os
import argparse
import random
import numpy as np
from PIL import Image
from utils import read_color_mapping  # your provided mapping functions

# ----------------------------
# Check if an RGB image contains a specific color
# ----------------------------
def contains_color(img, color):
    """
    Checks whether the given PIL RGB image contains at least one pixel equal to 'color'.
    """
    arr = np.array(img)  # shape: (H, W, 3)
    return np.any(np.all(arr == color, axis=-1))

# ----------------------------
# Scale and Preserve Original Size
# ----------------------------
def scale_and_preserve_size(image, label, scale_factor):
    """
    Scales the image and label by scale_factor and then adjusts back to the original size.
    If the scaled image is larger, it is center-cropped.
    If it is smaller, it is center-padded (using background color (0,0,0)).
    
    Both image and label are PIL Images.
    """
    orig_w, orig_h = image.size
    new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
    
    image_scaled = image.resize((new_w, new_h), Image.BILINEAR)
    label_scaled = label.resize((new_w, new_h), Image.NEAREST)
    
    if new_w >= orig_w and new_h >= orig_h:
        left = (new_w - orig_w) // 2
        top = (new_h - orig_h) // 2
        image_final = image_scaled.crop((left, top, left + orig_w, top + orig_h))
        label_final = label_scaled.crop((left, top, left + orig_w, top + orig_h))
    else:
        image_final = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
        label_final = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
        offset_x = (orig_w - new_w) // 2
        offset_y = (orig_h - new_h) // 2
        image_final.paste(image_scaled, (offset_x, offset_y))
        label_final.paste(label_scaled, (offset_x, offset_y))
    
    return image_final, label_final

# ----------------------------
# Random Crop, Flip, and Preserve Original Size
# ----------------------------
def random_crop_flip(image, label, crop_ratio=0.8):
    """
    Randomly crops a region (crop_ratio of original dimensions) from the image and label,
    randomly flips horizontally, and then pads back to the original size (centered).
    Both image and label are PIL Images.
    """
    orig_w, orig_h = image.size
    crop_w, crop_h = int(orig_w * crop_ratio), int(orig_h * crop_ratio)
    if crop_w < 1 or crop_h < 1:
        return image, label
    
    left = random.randint(0, orig_w - crop_w)
    top = random.randint(0, orig_h - crop_h)
    right, bottom = left + crop_w, top + crop_h
    
    image_cropped = image.crop((left, top, right, bottom))
    label_cropped = label.crop((left, top, right, bottom))
    
    if random.random() > 0.5:
        image_cropped = image_cropped.transpose(Image.FLIP_LEFT_RIGHT)
        label_cropped = label_cropped.transpose(Image.FLIP_LEFT_RIGHT)
    
    image_final = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
    label_final = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
    offset_x = (orig_w - crop_w) // 2
    offset_y = (orig_h - crop_h) // 2
    image_final.paste(image_cropped, (offset_x, offset_y))
    label_final.paste(label_cropped, (offset_x, offset_y))
    
    return image_final, label_final

# ----------------------------
# Get Bounding Box for a Target Color in an RGB Label
# ----------------------------
def get_bounding_box(label, target_color):
    """
    Finds the bounding box (left, top, right, bottom) for pixels in the label equal to target_color.
    Returns None if no such pixels are found.
    """
    arr = np.array(label)
    mask = np.all(arr == target_color, axis=-1)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    return (min_x, min_y, max_x, max_y)

# ----------------------------
# Object-Centered Crop Augmentation
# ----------------------------
def object_centered_crop(image, label, target_color, crop_scales=[1.0, 1.2, 1.5]):
    """
    Crops the image and label around the center of the object with target_color.
    The crop size is determined by the bounding box of the object multiplied by each factor in crop_scales.
    The crop is then resized back to the original image size.
    Both image and label are PIL Images.
    
    Returns a list of tuples: (suffix, augmented_image, augmented_label)
    including both flipped and non-flipped versions.
    """
    bbox = get_bounding_box(label, target_color)
    if bbox is None:
        return []
    min_x, min_y, max_x, max_y = bbox
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    obj_width = max_x - min_x
    obj_height = max_y - min_y
    orig_w, orig_h = image.size
    
    results = []
    for scale in crop_scales:
        crop_w = int(obj_width * scale)
        crop_h = int(obj_height * scale)
        left = max(0, center_x - crop_w // 2)
        top = max(0, center_y - crop_h // 2)
        right = left + crop_w
        bottom = top + crop_h
        if right > orig_w:
            right = orig_w
            left = right - crop_w
        if bottom > orig_h:
            bottom = orig_h
            top = bottom - crop_h
        crop_box = (left, top, right, bottom)
        cropped_img = image.crop(crop_box)
        cropped_label = label.crop(crop_box)
        resized_img = cropped_img.resize((orig_w, orig_h), Image.BILINEAR)
        resized_label = cropped_label.resize((orig_w, orig_h), Image.NEAREST)
        results.append((f"objcrop{scale}", resized_img, resized_label))
        flipped_img = resized_img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_label = resized_label.transpose(Image.FLIP_LEFT_RIGHT)
        results.append((f"objcrop{scale}_flip", flipped_img, flipped_label))
    return results

# ----------------------------
# Main Procedure
# ----------------------------
def main():
    image_dir = '../data/CamVid/train'
    label_dir = '../data/CamVid/train_labels'
    csv_path = '../data/CamVid/class_dict.csv'
    
    # Read the color mapping from the CSV.
    mapping, mapping_inv = read_color_mapping(csv_path)
    
    # List all image files.
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        base_name, ext = os.path.splitext(img_file)
        label_file = f"{base_name}_L.png"
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            print(f"Warning: Label for {img_file} not found, skipping.")
            continue
        
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        
        # Check for the presence of specific classes.
        has_pedestrian = contains_color(label, mapping.get(2, (0,0,0)))
        has_bicyclist = contains_color(label, mapping.get(3, (0,0,0)))
        has_motorcycle = contains_color(label, mapping.get(4, (0,0,0)))
        has_truck_bus = contains_color(label, mapping.get(5, (0,0,0)))
        
        aug_results = []  # List to store (suffix, augmented_image, augmented_label)
        
        # Option 1: Multiple scaling factors for Pedestrian or Bicyclist.
        if has_pedestrian or has_bicyclist or has_motorcycle:
            for factor in [1.2, 1.7, 2.2]:
                up_img, up_label = scale_and_preserve_size(image, label, scale_factor=factor)
                aug_results.append((f"upscaled_{factor}", up_img, up_label))
        
        # Option 2: Multiple downscale factors for Truck_Bus.
        # if has_truck_bus:
        #     for factor in [0.8, 0.5]:
        #         down_img, down_label = scale_and_preserve_size(image, label, scale_factor=factor)
        #         aug_results.append((f"downscaled_{factor}", down_img, down_label))
        
        # Option 3: Object-centered crop around low-accuracy classes (MotorcycleScooter and Truck_Bus).
        if has_motorcycle or has_truck_bus:
            for target in [4, 5]:
                target_color = mapping.get(target, (0,0,0))
                crops = object_centered_crop(image, label, target_color, crop_scales=[1.5, 2, 3])
                for suffix, crop_img, crop_lbl in crops:
                    aug_results.append((f"{target}_{suffix}", crop_img, crop_lbl))
        
        # Option 4: Random crop and flip (as an additional augmentation).
        if has_motorcycle or has_truck_bus:
            rc_img, rc_label = random_crop_flip(image, label, crop_ratio=0.8)
            aug_results.append(("randcropflip", rc_img, rc_label))
        
        # Save all augmented versions.
        for suffix, aug_img, aug_label in aug_results:
            out_img_name = f"{base_name}_{suffix}{ext}"
            out_label_name = f"{base_name}_{suffix}_L.png"
            aug_img.save(os.path.join(image_dir, out_img_name))
            aug_label.save(os.path.join(label_dir, out_label_name))
            print(f"Saved {out_img_name} and {out_label_name}")

if __name__ == "__main__":
    main()