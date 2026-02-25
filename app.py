from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import json
import io

app = Flask(__name__)

# ── Build model architecture ──
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ── Load everything at startup ──
print("Loading model...")
with open('class_names.json') as f:
    class_names = json.load(f)

model = build_model(26)   # ← hardcode 26 to match trained weights
model.load_weights('food_model.weights.h5')
print(f"✅ Ready! {len(class_names)} classes loaded.")

with open('ingredients.json') as f:
    ingredients_db = json.load(f)

with open('allergens.json') as f:
    allergens_db = json.load(f)


# ── Helper: preprocess image ──
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ── Helper: rule engine ──
def check_allergens(dish_name, user_allergens):
    dish_ingredients = ingredients_db.get(dish_name, [])

    triggered = []
    for allergen in user_allergens:
        trigger_ingredients = allergens_db.get(allergen, [])
        for ing in dish_ingredients:
            if ing in trigger_ingredients:
                triggered.append({
                    "allergen": allergen,
                    "found_ingredient": ing
                })
                break

    count = len(triggered)
    if count == 0:
        risk = "SAFE"
    elif count == 1:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "dish": dish_name,
        "ingredients": dish_ingredients,
        "triggered": triggered,
        "risk_level": risk
    }


# ── Routes ──
@app.route('/')
def home():
    return render_template('landing.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Check allergens were sent
    user_allergens = request.form.getlist('allergens')
    if not user_allergens:
        return jsonify({"error": "No allergens selected"}), 400

    # Read and preprocess image
    image_bytes = request.files['image'].read()
    img_array = preprocess_image(image_bytes)

    # Predict dish
    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    dish_name = class_names[predicted_index]

    # Get top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3 = [
        {
            "dish": class_names[i],
            "confidence": round(float(predictions[0][i]) * 100, 1)
        }
        for i in top3_indices
    ]

    # Run allergen check
    result = check_allergens(dish_name, user_allergens)

    return jsonify({
        "dish": dish_name,
        "confidence": round(confidence, 1),
        "top3": top3,
        "ingredients": result["ingredients"],
        "triggered": result["triggered"],
        "risk_level": result["risk_level"]
    })


if __name__ == '__main__':
    app.run(debug=True)
