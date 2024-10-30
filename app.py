from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import os
import gc

from ultralytics import YOLO

model = YOLO('best.pt')

#Color map for 5 classes
COLOR_MAP = {
    0: (255, 0, 0),      # Red
    1: (0, 255, 0),      # Green
    2: (0, 150, 255),    # Blue
    3: (255, 255, 0),    # Yellow
    4: (255, 0, 255),    # Magenta
    5: (0, 255, 255),    # Cyan
    6: (128, 0, 128),    # Purple
    7: (255, 165, 0),    # Orange
    8: (128, 128, 0)     # Olive
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


CLASS_NAMES = ["Candida Albicans","Enterococcus Faecalis","Escherichia Coli", "Klebsiella Pneumoniae", "Pseudomonas Aeruginosa", "Staphylococcus Aureusr","Staphylococcus Epidermidis", "Staphylococcus Saprophyticus","Streptococcus Agalactiae","Background"]


def predict_on_image(image_stream):
    gc.collect()
    # Read the image from the stream
    image_bytes = image_stream.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(image, conf=0.5)

    segmented_image = image.copy()
    class_info = []

    for r in results:
        if r.masks is not None:
            for seg, cls, conf in zip(r.masks.data, r.boxes.cls, r.boxes.conf):
                # Convert the mask to a binary image
                mask = seg.cpu().numpy().astype(np.uint8) * 255
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

                # Get the color, class name, and confidence for this prediction
                class_idx = int(cls.item())
                color = COLOR_MAP[class_idx]
                class_name = CLASS_NAMES[class_idx]
                confidence = conf.item()

                # Create a colored mask
                color_mask = np.zeros_like(segmented_image)
                color_mask[mask > 0] = color

                # Blend the color mask with the original image
                alpha = 0.5  # Transparency of the mask
                segmented_image = cv2.addWeighted(segmented_image, 1, color_mask, alpha, 0)

                # Store class info
                class_info.append(f"{class_name}: {confidence:.2f}")

    # Encode images to base64
    _, buffer_original = cv2.imencode('.png', image)
    original_img_base64 = base64.b64encode(buffer_original).decode('utf-8')

    _, buffer_segmented = cv2.imencode('.png', segmented_image)
    detection_img_base64 = base64.b64encode(buffer_segmented).decode('utf-8')
    gc.collect()
    return original_img_base64, detection_img_base64, class_info


@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def home():
    gc.collect()
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            original_img_base64, detection_img_base64, class_info = predict_on_image(file.stream)

            return render_template('result.html',
                                   original_img_data=original_img_base64,
                                   detection_img_data=detection_img_base64,
                                   class_info=class_info)

    return render_template('index.html')


if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=5000, host='0.0.0.0')


