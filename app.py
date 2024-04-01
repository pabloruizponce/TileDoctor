from ultralytics import YOLO
import gradio as gr
import numpy as np
import cv2

# Define the prediction function
def predict(image):
    # Perform inference with your model
    results = model(image)[0]
    # Initialize a dictionary to hold combined masks for each class
    combined_masks = {}
    class_labels = {}

    # Load your image to get its dimensions
    height, width = image.shape[:2]

    # Iterate over each detected object
    for idx, segment in enumerate(results.masks.xy):
        # Create an empty mask with the same dimensions as the image
        mask = np.zeros((height, width), dtype=np.int32)

        # Assuming segment is a list of (x, y) tuples
        segment_np = np.array([segment], dtype=np.int32)

        cv2.fillPoly(mask, pts=segment_np, color=(255))
        mask = (mask / 255.0).astype(np.float32)

        cls = results.boxes.cls[idx]
        conf = results.boxes.conf[idx]
        label = model.names[int(cls)]  # Assuming model.names contains class labels

        # If the class label is already in the dictionary, combine the masks
        if label in combined_masks:
            combined_masks[label] += mask
            combined_masks[label] = np.clip(combined_masks[label], 0, 1)  # Ensure mask values are within [0, 1]
        else:
            combined_masks[label] = mask
            class_labels[label] = label

    # Prepare the annotations list
    annots = []
    for label, mask in combined_masks.items():
        annots.append((mask, label))

    return image, annots


if __name__ == '__main__':
    # Load your trained YOLOv8 model
    model = YOLO('model.pt')

    # Create the Gradio interface
    iface = gr.Interface(
        fn=predict,
        inputs="image",
        outputs="annotated_image"
    )

    # Launch the demo
    iface.launch(debug=True)