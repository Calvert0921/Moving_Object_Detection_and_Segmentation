import os
import cv2
import numpy as np

def find_images_with_color(directory, target_color=(64, 0, 128)):
    found_images = []
    count = 0
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):  # Check for common image file types
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Skipping {filename}: Unable to read the image.")
                continue
            
            # Convert BGR (OpenCV default) to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Check if the color exists in the image
            mask = np.all(image_rgb == target_color, axis=-1)
            
            if np.any(mask):
                found_images.append(filename)
                print(f"Color found in: {filename}")
                count += 1
    
    if not found_images:
        print("No images contain the target color.")
    
    print(f"Number of images: {count}")
    return found_images

# Example usage
directory_path = "data/CamVid/train_labels"  # Change this to your actual directory
find_images_with_color(directory_path)