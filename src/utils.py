import numpy as np
import csv

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