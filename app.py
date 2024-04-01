import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import cv2
import io  

from ultralytics import YOLO
from PIL import Image

# New function to generate a bar plot and return it as an image
def generate_bar_plot(class_percentages):
    # Calculate the total percentage covered by detected classes
    total_covered_percentage = sum(class_percentages.values())
    # Calculate the background percentage
    background_percentage = 100 - total_covered_percentage
    
    # Add the background class to the class_percentages dictionary
    class_percentages['Background'] = background_percentage
    
    # Sizes for each slice
    sizes = list(class_percentages.values())
    # Labels for each slice
    labels = list(class_percentages.keys())
    # Colors for each slice, adding one more color for the background
    slice_colors = [colors[label] for label in labels]

    # Creating the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=slice_colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    # Convert buffer to image
    pie_image = Image.open(buf)
    # Convert image to numpy array
    pie_image = np.array(pie_image)
    plt.close()  # Close the plot to free memory
    
    return pie_image


# Define the prediction function
def predict(image):

    # Perform inference with your model
    results = model(image)[0]
    # Initialize a dictionary to hold combined masks for each class
    combined_masks = {}
    class_labels = {}

    # Load your image to get its dimensions
    height, width = image.shape[:2]

    # Convert image to grayscale to simplify background detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find all pixels that are not 255 (not background)
    total_area = np.count_nonzero(gray_image != 255)

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
        label = names[int(cls)]  # Assuming model.names contains class labels

        # If the class label is already in the dictionary, combine the masks
        if label in combined_masks:
            combined_masks[label] += mask
            combined_masks[label] = np.clip(combined_masks[label], 0, 1)  # Ensure mask values are within [0, 1]
        else:
            combined_masks[label] = mask
            class_labels[label] = label

    # Calculate the percentage of each class
    class_percentages = {}
    for label, mask in combined_masks.items():
        class_area = np.sum(mask)
        class_percentages[label] = (class_area / total_area) * 100

    plot = generate_bar_plot(class_percentages)

    # Prepare the annotations list
    annots = []
    for label, mask in combined_masks.items():
        annots.append((mask, label))

    return (image, annots), plot

if __name__ == '__main__':
    # Load your trained YOLOv8 model
    model = YOLO('model.pt')
    names = {0: 'Xanthoria', 1: 'Moho', 2: 'Caloplaca', 3: 'Verrucaria Nigrescens'}
    colors = {'Xanthoria':"#1f77b4", 'Moho':"#ff7f0e", 'Caloplaca':"#2ca02c", 'Verrucaria Nigrescens':"#d62728", 'Background':"#e3e3e3"}  # Blue, Orange, Green, Red

    # Create the Gradio interface
    iface = gr.Interface(
        fn=predict,
        inputs="image",
        outputs=[gr.AnnotatedImage(color_map=colors), "image"]
    )

    # Launch the demo
    iface.launch()