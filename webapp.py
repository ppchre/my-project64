import os
import cv2 
import numpy as np
from mtcnn import MTCNN
from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import random

# import os

# Generate a random secret key
secret_key = os.urandom(24).hex()

app = Flask(__name__)
app.secret_key = secret_key 

# Path to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model and face detector initialization
model = load_model('facial_modelling.h5')
detector = MTCNN()

# Image size as expected by the model
image_size = (256, 256)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)
    
def preprocess_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise Exception("Could not read the image.")

    # Convert to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)

    if results:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = image_rgb[y1:y2, x1:x2]
        face_array = cv2.resize(face, image_size)

        mask = np.zeros((image_size[0], image_size[1]), dtype="uint8")
        resized_x1, resized_y1, resized_x2, resized_y2 = get_resized_coordinates(x1, y1, width, height, image_rgb)
        cv2.rectangle(mask, (resized_x1, resized_y1), (resized_x2, resized_y2), 255, -1)

        blurred_image = cv2.GaussianBlur(face_array, (21, 21), 0)
        final_image = cv2.bitwise_and(blurred_image, blurred_image, mask=cv2.bitwise_not(mask))
        final_image += cv2.bitwise_and(face_array, face_array, mask=mask)

        final_image_preprocessed = preprocess_input(final_image)  # Preprocess for ResNet50
        return final_image_preprocessed
    else:
        raise Exception("No face detected in the image.")

def get_resized_coordinates(x1, y1, width, height, original_image):
    resized_x1 = int(image_size[0] * x1 / original_image.shape[1])
    resized_y1 = int(image_size[1] * y1 / original_image.shape[0])
    resized_x2 = resized_x1 + int(image_size[0] * width / original_image.shape[1])
    resized_y2 = resized_y1 + int(image_size[1] * height / original_image.shape[0])
    return resized_x1, resized_y1, resized_x2, resized_y2

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(file_path)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)

        # Map the predicted class index to your class names with " Face" appended
        class_names = ['Diamond Face', 'Heart Face', 'Long Face', 'Oval Face', 'Round Face', 'Square Face', 'Triangle Face']
        predicted_class_name = class_names[predicted_class[0]]

        # Store the processed image URL in the session
        session['processed_image_url'] = file_path  # You can use a URL instead of the file path if needed
    except Exception as e:
        print(f"Error in processing the image: {str(e)}")
        return f"Error in processing the image: {str(e)}"


    event_type = request.form['event_type']
    age = request.form['age']  # Keep 'age' as a string since it can be "15 - 59" or "60 ปีขึ้นไป"
    thai_dress = 'ไทย' if 'thai_dress' in request.form and request.form['thai_dress'] == 'Yes' else ''

    # Since 'age' is a string, directly use it as 'age_group'
    age_group = age if event_type in ["งานมงคล (ทำบุญ, งานบวช)", "งานอวมงคล (งานศพ)", "งานเลี้ยงกลางคืน"] else ""

    # Debug prints
    print(f"Event Type: {event_type}")
    print(f"Age: {age}")
    print(f"Thai Dress: {thai_dress}")
    print(f"Predicted Class Name: {predicted_class_name}")
    print(f"Age Group: {age_group}")

    if event_type == "งานแต่ง (เจ้าสาว)" and thai_dress:
        image_dir = os.path.join('Type of Events', event_type, predicted_class_name, thai_dress)
    elif event_type == "งานมงคล":
        image_dir = os.path.join('Type of Events', event_type, predicted_class_name)
    elif event_type == "งานอวมงคล (งานศพ)":
        # For "งานอวมงคล (งานศพ)", do not include age_group in the path
        image_dir = os.path.join('Type of Events', event_type, predicted_class_name)
    else:
        if age_group:
            image_dir = os.path.join('Type of Events', event_type, predicted_class_name, age_group)
        else:
            image_dir = os.path.join('Type of Events', event_type, predicted_class_name)

    # List and randomly select up to 3 images from the directory
    try:
        full_image_dir = os.path.join(app.static_folder, image_dir)
        print(f"Full Image Directory: {full_image_dir}")
        if not os.path.isdir(full_image_dir):
            return f"Directory does not exist: {full_image_dir}"

        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')  # Add or remove extensions as needed
        image_files = [f for f in os.listdir(full_image_dir) if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(full_image_dir, f))]

        if not image_files:
            return "No valid images found in the directory."

        selected_images = random.sample(image_files, min(len(image_files), 3))
        selected_images_paths = [os.path.join(image_dir, img) for img in selected_images]
    except Exception as e:
        print(f"Error in selecting recommendations: {str(e)}")
        return f"Error in selecting recommendations: {str(e)}"
    selected_images_paths = [path.replace('\\', '/') for path in selected_images_paths]
    print(f"Selected Images Paths: {selected_images_paths}")

        # Render a template to display the selected images
    return render_template('recommendations.html', images=selected_images_paths, predicted_classes=predicted_class_name, processed_image_url=session.get('processed_image_url', ''))

if __name__ == '__main__':
    app.run(debug=True)
