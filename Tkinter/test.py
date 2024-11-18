<<<<<<< HEAD
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import os

# Load the YOLOv5 model
model_path = "C:/PD/Tkinter/yolov5/runs/train/custom_model/weights/best.onnx"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Temporary file for captured image
temp_file = "temp_captured_image.jpg"

# Function to handle image capture and display
def capture_image():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera, change if using another device

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Update button text to indicate capturing
    capture_button.config(text="Capturing...", state="disabled", bg="#ffcc00")
    root.update_idletasks()

    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Save the captured frame to a temporary file
        cv2.imwrite(temp_file, frame)

        # Release the webcam
        cap.release()
        cv2.destroyAllWindows()

        # Open and resize the captured image
        img = Image.open(temp_file)
        img = img.resize((300, 300))  # Resize to fit in the larger display area
        img_tk = ImageTk.PhotoImage(img)

        # Clear previous images and update the captured image on the left side
        uploaded_image_label.config(image=img_tk)
        uploaded_image_label.image = img_tk  # Keep a reference to prevent garbage collection

        # Reset detection result before performing new detection
        result_image_label.config(image="", text="Result Image will be displayed here")
        result_image_label.image = None
        angle_text_label.config(text="Detected Angle: N/A")

        # Perform detection using YOLOv5
        detect_and_display(temp_file)

        # Remove temporary file after use
        if os.path.exists(temp_file):
            os.remove(temp_file)
    else:
        print("Error: Unable to capture image.")
        cap.release()
        cv2.destroyAllWindows()

    # Restore the button state
    capture_button.config(text="Capture Image", state="normal", bg="#00509e")

# Function to perform detection and display results without saving
def detect_and_display(file_path):
    # Perform inference on the captured image
    results = model(file_path)

    # Extract the detection result
    angle_detected = "N/A"  # Initialize to N/A
    if len(results.xyxy[0]) > 0:  # Check if detections exist
        # Assume the angle is in the label (adjust based on your model's label format)
        angle_detected = results.pandas().xyxy[0]['name'][0]  # First detection

    # Update detected angle text
    angle_text_label.config(text=f"Detected Angle: {angle_detected}")

    # Display the detected result directly from results object
    results_img = results.render()[0]  # Render the result image in memory
    detected_img = Image.fromarray(results_img)  # Convert to PIL Image for Tkinter
    detected_img = detected_img.resize((300, 300))  # Resize to fit the larger display area
    detected_img_tk = ImageTk.PhotoImage(detected_img)

    # Display the detected result image on the right side
    result_image_label.config(image=detected_img_tk, text="")  # Remove placeholder text when showing image
    result_image_label.image = detected_img_tk  # Keep a reference to prevent garbage collection

# Set up the main window
root = tk.Tk()
root.title("Image Angle Detection")
root.geometry("800x480")
root.configure(bg="#f4f4f4")

# Left frame
left_frame = tk.Frame(root, bg="#e6f0fa", width=380, height=460)
left_frame.pack_propagate(False)
left_frame.pack(side="left", padx=(10, 5), pady=10, fill="y")

upload_header = tk.Label(left_frame, text="CAPTURE IMAGE", font=("Helvetica", 16, "bold"), bg="#e6f0fa", fg="#003366")
upload_header.pack(pady=(10, 10))

uploaded_image_border = tk.Frame(left_frame, bg="#003366", width=300, height=300)
uploaded_image_border.pack(pady=(10, 10))
uploaded_image_label = tk.Label(uploaded_image_border, bg="#e6f0fa")
uploaded_image_label.pack()

left_frame_spacer = tk.Frame(left_frame, bg="#e6f0fa", height=30)
left_frame_spacer.pack(fill="y", expand=True)

capture_button = tk.Button(
    left_frame, text="Capture Image", command=capture_image,
    bg="#00509e", fg="white", font=("Helvetica", 12, "bold"), width=18, height=2,
    relief="flat", activebackground="#003366", activeforeground="white", bd=2
)
capture_button.pack(pady=(5, 10), side="bottom")

separator = ttk.Separator(root, orient="vertical")
separator.pack(side="left", fill="y", padx=(0, 5))

# Right frame
right_frame = tk.Frame(root, bg="#e6f0fa", width=380, height=460)
right_frame.pack_propagate(False)
right_frame.pack(side="right", padx=(5, 10), pady=10, fill="y")

detected_header = tk.Label(right_frame, text="DETECTION RESULT", font=("Helvetica", 16, "bold"), bg="#e6f0fa", fg="#003366")
detected_header.pack(pady=(10, 10))

result_image_border = tk.Frame(right_frame, bg="#003366", width=300, height=300, bd=2, relief="solid")
result_image_border.pack(pady=(10, 10))
result_image_label = tk.Label(
    result_image_border,
    text="Result Image will be displayed here",
    font=("Helvetica", 12, "bold", "italic"),
    bg="#003366", fg="#ffffff",
    relief="flat", wraplength=280,
    justify="center", padx=10, pady=10
)
result_image_label.pack(fill="both", expand=True)

right_frame_spacer = tk.Frame(right_frame, bg="#e6f0fa", height=30)
right_frame_spacer.pack(fill="y", expand=True)

angle_text_label = tk.Label(
    right_frame,
    text="Detected Angle: N/A",
    font=("Helvetica", 14, "bold"),
    bg="#00509e", fg="white",
    width=25, height=2,
    relief="flat"
)
angle_text_label.pack(pady=(5, 10), side="bottom")

root.mainloop()
=======
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import os

# Load the YOLOv5 model
model_path = "C:/PD/Tkinter/yolov5/runs/train/custom_model/weights/best.onnx"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Temporary file for captured image
temp_file = "temp_captured_image.jpg"

# Function to handle image capture and display
def capture_image():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera, change if using another device

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Update button text to indicate capturing
    capture_button.config(text="Capturing...", state="disabled", bg="#ffcc00")
    root.update_idletasks()

    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Save the captured frame to a temporary file
        cv2.imwrite(temp_file, frame)

        # Release the webcam
        cap.release()
        cv2.destroyAllWindows()

        # Open and resize the captured image
        img = Image.open(temp_file)
        img = img.resize((300, 300))  # Resize to fit in the larger display area
        img_tk = ImageTk.PhotoImage(img)

        # Clear previous images and update the captured image on the left side
        uploaded_image_label.config(image=img_tk)
        uploaded_image_label.image = img_tk  # Keep a reference to prevent garbage collection

        # Reset detection result before performing new detection
        result_image_label.config(image="", text="Result Image will be displayed here")
        result_image_label.image = None
        angle_text_label.config(text="Detected Angle: N/A")

        # Perform detection using YOLOv5
        detect_and_display(temp_file)

        # Remove temporary file after use
        if os.path.exists(temp_file):
            os.remove(temp_file)
    else:
        print("Error: Unable to capture image.")
        cap.release()
        cv2.destroyAllWindows()

    # Restore the button state
    capture_button.config(text="Capture Image", state="normal", bg="#00509e")

# Function to perform detection and display results without saving
def detect_and_display(file_path):
    # Perform inference on the captured image
    results = model(file_path)

    # Extract the detection result
    angle_detected = "N/A"  # Initialize to N/A
    if len(results.xyxy[0]) > 0:  # Check if detections exist
        # Assume the angle is in the label (adjust based on your model's label format)
        angle_detected = results.pandas().xyxy[0]['name'][0]  # First detection

    # Update detected angle text
    angle_text_label.config(text=f"Detected Angle: {angle_detected}")

    # Display the detected result directly from results object
    results_img = results.render()[0]  # Render the result image in memory
    detected_img = Image.fromarray(results_img)  # Convert to PIL Image for Tkinter
    detected_img = detected_img.resize((300, 300))  # Resize to fit the larger display area
    detected_img_tk = ImageTk.PhotoImage(detected_img)

    # Display the detected result image on the right side
    result_image_label.config(image=detected_img_tk, text="")  # Remove placeholder text when showing image
    result_image_label.image = detected_img_tk  # Keep a reference to prevent garbage collection

# Set up the main window
root = tk.Tk()
root.title("Image Angle Detection")
root.geometry("800x480")
root.configure(bg="#f4f4f4")

# Left frame
left_frame = tk.Frame(root, bg="#e6f0fa", width=380, height=460)
left_frame.pack_propagate(False)
left_frame.pack(side="left", padx=(10, 5), pady=10, fill="y")

upload_header = tk.Label(left_frame, text="CAPTURE IMAGE", font=("Helvetica", 16, "bold"), bg="#e6f0fa", fg="#003366")
upload_header.pack(pady=(10, 10))

uploaded_image_border = tk.Frame(left_frame, bg="#003366", width=300, height=300)
uploaded_image_border.pack(pady=(10, 10))
uploaded_image_label = tk.Label(uploaded_image_border, bg="#e6f0fa")
uploaded_image_label.pack()

left_frame_spacer = tk.Frame(left_frame, bg="#e6f0fa", height=30)
left_frame_spacer.pack(fill="y", expand=True)

capture_button = tk.Button(
    left_frame, text="Capture Image", command=capture_image,
    bg="#00509e", fg="white", font=("Helvetica", 12, "bold"), width=18, height=2,
    relief="flat", activebackground="#003366", activeforeground="white", bd=2
)
capture_button.pack(pady=(5, 10), side="bottom")

separator = ttk.Separator(root, orient="vertical")
separator.pack(side="left", fill="y", padx=(0, 5))

# Right frame
right_frame = tk.Frame(root, bg="#e6f0fa", width=380, height=460)
right_frame.pack_propagate(False)
right_frame.pack(side="right", padx=(5, 10), pady=10, fill="y")

detected_header = tk.Label(right_frame, text="DETECTION RESULT", font=("Helvetica", 16, "bold"), bg="#e6f0fa", fg="#003366")
detected_header.pack(pady=(10, 10))

result_image_border = tk.Frame(right_frame, bg="#003366", width=300, height=300, bd=2, relief="solid")
result_image_border.pack(pady=(10, 10))
result_image_label = tk.Label(
    result_image_border,
    text="Result Image will be displayed here",
    font=("Helvetica", 12, "bold", "italic"),
    bg="#003366", fg="#ffffff",
    relief="flat", wraplength=280,
    justify="center", padx=10, pady=10
)
result_image_label.pack(fill="both", expand=True)

right_frame_spacer = tk.Frame(right_frame, bg="#e6f0fa", height=30)
right_frame_spacer.pack(fill="y", expand=True)

angle_text_label = tk.Label(
    right_frame,
    text="Detected Angle: N/A",
    font=("Helvetica", 14, "bold"),
    bg="#00509e", fg="white",
    width=25, height=2,
    relief="flat"
)
angle_text_label.pack(pady=(5, 10), side="bottom")

root.mainloop()
>>>>>>> 1b550bc (project)
