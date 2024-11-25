import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2, Preview
from time import sleep
import RPi.GPIO as GPIO

# Global variables
totalTime = 30  # Total time for one complete cycle (in seconds)

# GPIO pin setup
road2Green = 17
road2Yellow = 27
road2Red = 22

road1Green = 5
road1Yellow = 6
road1Red = 13

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup([road2Green, road2Yellow, road2Red, road1Green, road1Yellow, road1Red], GPIO.OUT, initial=GPIO.LOW)

# Function to control traffic lights
def controlTrafficLights(highDensityGreen, highDensityYellow, highDensityRed,
                         lowDensityGreen, lowDensityYellow, lowDensityRed,
                         highDensityTime, lowDensityTime):
    # Both lights red initially
    GPIO.output([highDensityRed, lowDensityRed], GPIO.HIGH)
    sleep(2)  # Initial safety delay

    # High-density road starts: Red → Yellow → Green
    GPIO.output(highDensityRed, GPIO.LOW)  # Turn off red
    GPIO.output(highDensityYellow, GPIO.HIGH)  # Turn on yellow
    sleep(highDensityTime / 3)  # Yellow duration

    GPIO.output(highDensityYellow, GPIO.LOW)  # Turn off yellow
    GPIO.output(highDensityGreen, GPIO.HIGH)  # Turn on green
    sleep(highDensityTime * (2 / 3))  # Green duration

    # While high-density road green is ending, low-density road yellow starts
    GPIO.output(lowDensityRed, GPIO.LOW)  # Turn off red of low-density road
    GPIO.output(lowDensityYellow, GPIO.HIGH)  # Low-density yellow on
    sleep(highDensityTime / 3)  # Overlap yellow duration for low-density

    # High-density green ends, turn red
    GPIO.output(highDensityGreen, GPIO.LOW)  # Turn off green
    GPIO.output(highDensityRed, GPIO.HIGH)  # Turn on red

    # Low-density road: Yellow → Green
    GPIO.output(lowDensityYellow, GPIO.LOW)  # Turn off yellow
    GPIO.output(lowDensityGreen, GPIO.HIGH)  # Turn on green
    sleep(lowDensityTime * (2 / 3))  # Green duration

    # Low-density green ends: Green → Yellow → Red
    GPIO.output(lowDensityGreen, GPIO.LOW)  # Turn off green
    GPIO.output(lowDensityYellow, GPIO.HIGH)  # Turn on yellow
    sleep(lowDensityTime / 3)  # Yellow duration
    GPIO.output(lowDensityYellow, GPIO.LOW)  # Turn off yellow
    GPIO.output(lowDensityRed, GPIO.HIGH)  # Turn on red

# Function to calculate time for each road based on density
def calculateTimes(road2Density, road1Density):
    totalDensity = road2Density + road1Density
    print(totalDensity)
    if totalDensity == 0:  # Avoid division by zero
        return totalTime // 2, totalTime // 2
    
    road2Ratio = road2Density / totalDensity
    road1Ratio = road1Density / totalDensity

    road2Time = round(totalTime * road2Ratio)
    road1Time = totalTime - road2Time  # Ensure total is exactly 60 seconds
    return road2Time, road1Time

# Main traffic light control function
def trafficControl(road2Density, road1Density):
    road2Time, road1Time = calculateTimes(road2Density, road1Density)
    print(f"Time allocated: Road 2: {road2Time}s, Road 1: {road1Time}s")
    # Decide which road has higher density and control accordingly
    if road2Density >= road1Density:
        # Road2 (higher density) goes first
        controlTrafficLights(
            road2Green, road2Yellow, road2Red,
            road1Green, road1Yellow, road1Red,
            road2Time, road1Time
        )
    else:
        # Road1 (higher density) goes first
        controlTrafficLights(
            road1Green, road1Yellow, road1Red,
            road2Green, road2Yellow, road2Red,
            road1Time, road2Time
        )

def capture_image():
    from picamera2 import Picamera2
    from time import sleep

    # Initialize and configure the camera
    camera = Picamera2()
    camera_config = camera.create_still_configuration()
    camera.configure(camera_config)

    try:
        # Start the camera
        camera.start()
        print("Camera started...")
        sleep(2)  # Allow time for adjustments

        # Capture the image
        image_path = "/home/vlsi/Desktop/final project sp/images/captured/image.jpg"
        camera.capture_file(image_path)
        print(f"Image captured and saved as {image_path}")
        return image_path

    finally:
        # Ensure the camera is properly stopped and released
        camera.stop()
        camera.close()
        print("Camera stopped and released.")

def load_and_run_tflite_model(tflite_model_path, image_path):
    """
    Load a YOLOv5 TFLite model, preprocess the image, and run inference.

    Parameters:
        tflite_model_path (str): Path to the YOLOv5 TFLite model.
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Raw output data from the model.
    """
    # Load the TFLite model
    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    image = Image.open(image_path).resize((640, 640))  # Resize for YOLOv5 model
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def non_max_suppression_numpy(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) using numpy.
    
    Parameters:
        boxes (numpy.ndarray): Array of bounding boxes [x_min, y_min, x_max, y_max].
        scores (numpy.ndarray): Array of confidence scores.
        iou_threshold (float): IOU threshold for NMS.
    
    Returns:
        list: Indices of the selected boxes after NMS.
    """
    indices = np.argsort(scores)[::-1]  # Sort by scores (descending)
    selected_indices = []

    while len(indices) > 0:
        current = indices[0]
        selected_indices.append(current)
        if len(indices) == 1:
            break

        current_box = boxes[current]
        rest_boxes = boxes[indices[1:]]

        # Compute IoU
        x_min = np.maximum(current_box[0], rest_boxes[:, 0])
        y_min = np.maximum(current_box[1], rest_boxes[:, 1])
        x_max = np.minimum(current_box[2], rest_boxes[:, 2])
        y_max = np.minimum(current_box[3], rest_boxes[:, 3])

        inter_area = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
        box_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        rest_area = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])

        union_area = box_area + rest_area - inter_area
        iou = inter_area / union_area

        # Suppress boxes with IoU greater than the threshold
        remaining_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining_indices + 1]

    return selected_indices

def process_and_draw_boxes(image_path, output_data, class_names, output_dir, threshold=0.5):
    """
    Process model output, draw bounding boxes, and annotate the image.

    Parameters:
        image_path (str): Path to the input image.
        output_data (numpy.ndarray): Raw output from the YOLOv5 TFLite model.
        class_names (list): List of class names.
        output_dir (str): Path to save the output image.
        threshold (float): Confidence threshold for filtering predictions.
    """
    # Load original image
    original_image = Image.open(image_path)
    image_width, image_height = original_image.size

    # Squeeze the batch dimension from the output
    output_data = np.squeeze(output_data)

    # Extract confidence scores and filter by threshold
    confidence_scores = output_data[:, 4]
    valid_indices = np.where(confidence_scores > threshold)
    filtered_predictions = output_data[valid_indices]

    # Extract bounding boxes and class probabilities
    boxes = filtered_predictions[:, :4]  # [x, y, width, height]
    class_probs = filtered_predictions[:, 5:]  # Class probabilities
    classes = np.argmax(class_probs, axis=-1)

    # Convert normalized box coordinates to pixel values
    boxes[:, [0, 2]] *= image_width / 640  # Scale x and width
    boxes[:, [1, 3]] *= image_height / 640  # Scale y and height
    boxes[:, 2] += boxes[:, 0]  # Convert width to x_max
    boxes[:, 3] += boxes[:, 1]  # Convert height to y_max

    # Non-Maximum Suppression (NMS)
    selected_indices = non_max_suppression_numpy(
        boxes, confidence_scores[valid_indices], iou_threshold=0.5
    )
    final_boxes = boxes[selected_indices]
    final_classes = classes[selected_indices]
    final_scores = confidence_scores[valid_indices][selected_indices]

    # Annotate image
    draw = ImageDraw.Draw(original_image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    vehicle_classes = {"car", "motorcycle", "bus", "truck"}
    vehicle_count = 0

    for i, box in enumerate(final_boxes):
        x_min, y_min, x_max, y_max = box
        class_name = class_names[final_classes[i]]
        label = f"{class_name} ({final_scores[i]:.2f})"

        if class_name in vehicle_classes:
            vehicle_count += 1

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

            # Draw label
            text_width, text_height = draw.textsize(label, font=font)
            draw.rectangle(
                [x_min, y_min - text_height, x_min + text_width, y_min],
                fill="red"
            )
            draw.text((x_min, y_min - text_height), label, fill="white", font=font)

    # Save annotated image
    output_image_path = f"{output_dir}/annotated_image.jpg"
    original_image.save(output_image_path)
    print(f"Annotated image saved at: {output_image_path}")
    print(f"Number of vehicles detected: {vehicle_count}")

    return vehicle_count


if __name__ == "__main__":
    try:
        GPIO.output([road1Red, road2Red], GPIO.HIGH)
        GPIO.output([road1Green, road1Yellow, road2Green, road2Yellow], GPIO.LOW)
        # Paths
        model_path = "/home/vlsi/Desktop/final project sp/yolov5s.tflite"
        output_dir = "/home/vlsi/Desktop/final project sp/output_images"

        # Class names (COCO dataset example)
        class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]

        # Capture and process the first image
        print("Capturing the first image...")
        input_image_path1 = capture_image()  # First capture
        print("Processing the first image...")
        output_data1 = load_and_run_tflite_model(model_path, input_image_path1)
        vehicle_count1 = process_and_draw_boxes(input_image_path1, output_data1, class_names, output_dir)
        print(f"Total vehicles detected in Image 1: {vehicle_count1}")

        # Properly release the camera and reinitialize for the second capture
        sleep(2)  # Small delay to ensure resources are freed

        # Capture and process the second image
        print("Capturing the second image...")
        input_image_path2 = capture_image()  # Second capture
        print("Processing the second image...")
        output_data2 = load_and_run_tflite_model(model_path, input_image_path2)
        vehicle_count2 = process_and_draw_boxes(input_image_path2, output_data2, class_names, output_dir)
        print(f"Total vehicles detected in Image 2: {vehicle_count2}")
        
        trafficControl(vehicle_count2, vehicle_count1)
        GPIO.output([road1Green, road1Yellow, road2Green, road2Yellow, road1Red, road2Red], GPIO.LOW)

    except Exception as e:
        print(f"An error occurred: {e}")
        GPIO.output([road1Green, road1Yellow, road2Green, road2Yellow, road1Red, road2Red], GPIO.LOW)

    finally:
        GPIO.output([road1Green, road1Yellow, road2Green, road2Yellow, road1Red, road2Red], GPIO.LOW)
        GPIO.cleanup()
        print("GPIO cleanup complete.")


    
