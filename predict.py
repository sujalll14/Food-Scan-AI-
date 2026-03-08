import os
import argparse
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

MODEL_NAME = "foodlens_model.h5"
CLASSES_FILE = "class_indices.json"
NUTRITION_FILE = "nutrition.json"

def predict_image(image_path):
    if not os.path.exists(MODEL_NAME):
        print(f"Error: Model {MODEL_NAME} not found. Please run train.py first.")
        sys.exit(1)
        
    if not os.path.exists(CLASSES_FILE):
        print(f"Error: {CLASSES_FILE} not found. Please run train.py first.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        sys.exit(1)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_NAME)
    
    with open(CLASSES_FILE, 'r') as f:
        class_indices = json.load(f)
        
    with open(NUTRITION_FILE, 'r') as f:
        nutrition_data = json.load(f)

    print(f"Processing image: {image_path}")
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array /= 255.0 # Rescale like in training

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_idx = str(tf.argmax(predictions[0]).numpy())
    
    predicted_class = class_indices[class_idx]
    confidence = 100 * predictions[0][int(class_idx)]
    
    print("\n" + "="*40)
    print(f"Prediction: {predicted_class.replace('_', ' ').title()}")
    print(f"Confidence: {confidence:.2f}%")
    
    if predicted_class in nutrition_data:
        nut_info = nutrition_data[predicted_class]
        print("\nNutritional Information (per 100g/standard serving):")
        print(f"  Calories: {nut_info['calories']} kcal")
        print(f"  Protein:  {nut_info['protein']}g")
        print(f"  Carbs:    {nut_info['carbs']}g")
    else:
        print("\nNo nutritional information available for this item.")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict food from an image using FoodLens")
    parser.add_argument('image_path', type=str, help='Path to the image to classify')
    args = parser.parse_args()
    predict_image(args.image_path)
