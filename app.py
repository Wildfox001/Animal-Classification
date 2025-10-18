from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("cattle_buffalo_model.h5")  # Make sure model is in same folder

# Mapping of classes
classes = {0: "buffalo", 1: "cattle"}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filepath = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(128, 128))
            x = image.img_to_array(img)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)

            pred = model.predict(x)
            label = classes[int(pred[0][0] > 0.5)]

            return render_template("result.html", filename=file.filename, label=label)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def send_file(filename):
    return f'<img src="/uploads/{filename}" alt="{filename}" width="300">'

if __name__ == "__main__":
    app.run(debug=True)
