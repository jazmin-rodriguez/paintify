from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from io import BytesIO
import base64
from sklearn.cluster import KMeans
import requests
from dotenv import load_dotenv

# Initialize the app and load environment variables
app = Flask(__name__)
load_dotenv()
api_key = os.getenv("DEEPAI_API_KEY")

# Helper functions
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_edge_detection(image, low_threshold=100, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def color_quantization(image, k=8):
    data = image.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    quantized_image = palette[labels.flatten()]
    quantized_image = quantized_image.reshape(image.shape).astype(np.uint8)
    return quantized_image


def paintify_image(image, edges, k=8):
    quantized_image = color_quantization(image, k)
    edges_blurred = cv2.medianBlur(edges, 5)
    edges_inv = cv2.bitwise_not(edges_blurred)
    edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
    paint_image = cv2.bitwise_and(quantized_image, edges_colored)
    paint_image = cv2.medianBlur(paint_image, 7)

    return paint_image

def preprocess_image(image_path):
    # Load and resize the image
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    image = cv2.resize(image, (500, 500))

    # Grayscale and blur for edge detection
    gray_image = convert_to_grayscale(image)
    blurred_gray = apply_gaussian_blur(gray_image)

    # Detect edges
    edges = apply_edge_detection(blurred_gray)

    # paintify the image
    paint_image = paintify_image(image, edges)

    return image, paint_image


def convert_image_to_base64(image):
    """Convert a cv2 image to a base64 string for JSON response."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_with_deepai(image_path, text_file):
    text_file = 'file.txt'
    """Send an image to the DeepAI API for further processing."""
    if not api_key:
        return {"error": "DeepAI API key is missing. Check your .env file."}
    
    try:
        with open(image_path, 'rb') as image_file:
            response = requests.post(
                "https://api.deepai.org/api/image-editor",
                files={
                    'image': image_file,
                    'text': (None, text_file),
                },
                headers={'api-key': api_key}
            )
        return response.json()
    except FileNotFoundError:
        return {"error": "Image file not found."}
    except Exception as e:
        return {"error": f"DeepAI API request failed: {str(e)}"}

# Routes
@app.route('/')
def upload_form():
    return render_template('interface.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        upload_folder = "uploads"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Process the image
        original_image, processed_image = preprocess_image(file_path)

        if original_image is None or processed_image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Convert images to base64 for JSON response
        original_image_base64 = convert_image_to_base64(original_image)
        processed_image_base64 = convert_image_to_base64(processed_image)

        # Remove the temporary file
        os.remove(file_path)

        return jsonify({
            "original_image": original_image_base64,
            "processed_image": processed_image_base64
        })

@app.route('/deepai', methods=['POST'])
def deepai_process():
    if 'file' not in request.files or 'text' not in request.form:
        return jsonify({"error": "Missing image or text description."}), 400

    file = request.files['file']
    text_description = request.form['text']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        upload_folder = "uploads"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the uploaded file
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Preprocess the image into a paintified version
        _, paint_image = preprocess_image(file_path)

        if paint_image is None:
            os.remove(file_path)
            return jsonify({"error": "Failed to process the image."}), 400

        # Convert the paintified image to a temporary in-memory file
        _, buffer = cv2.imencode('.png', paint_image)
        paint_image_file = BytesIO(buffer)

        # Send the paintified image to the DeepAI API
        try:
            response = requests.post(
                "https://api.deepai.org/api/image-editor",
                files={
                    'image': ('paint_image.png', paint_image_file, 'image/png'),
                    'text': (None, text_description),
                },
                headers={'api-key': api_key}
            )
            response_data = response.json()
        except Exception as e:
            os.remove(file_path)
            return jsonify({"error": f"DeepAI API request failed: {str(e)}"}), 500

        # Clean up the uploaded file
        os.remove(file_path)

        return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
