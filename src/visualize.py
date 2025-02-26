import cv2
import matplotlib.pyplot as plt

def visualize_bbox(image_path):
    # Load the original image
    image = cv2.imread(image_path)
    # Convert from BGR (OpenCV default) to RGB for proper display with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Example list of bounding boxes, where each entry is a dict containing a 'bbox' and a 'category_id'
    # For example: [{'bbox': [x, y, w, h], 'category_id': 2}, ...]
    # Replace this with the output from your extract_bounding_boxes function.
    boxes = [
        {'bbox': [389, 361, 395, 348], 'category': 'Car'}
    ]

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
