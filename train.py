import os
import zipfile
import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import argparse

# URL for a lightweight 10-class, 10% subset of Food-101
DATASET_URL = "https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip"
DATA_ZIP = "10_food_classes_10_percent.zip"
DATA_DIR = "10_food_classes_10_percent"
MODEL_NAME = "foodlens_model.h5"

def download_and_extract_data():
    if not os.path.exists(DATA_DIR):
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, DATA_ZIP)
        print("Extracting dataset...")
        with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
            zip_ref.extractall()
        print("Extraction complete.")
        os.remove(DATA_ZIP)
    else:
        print("Dataset already exists.")

def train_model(epochs):
    download_and_extract_data()

    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    # Image Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    print("Loading validation data...")
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Base model - MobileNetV2 is fast and lightweight
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base layers

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Added dropout for better generalization
    predictions = Dense(10, activation='softmax')(x) # 10 classes

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Starting initial training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )
    
    # UNFREEZE TOP LAYERS FOR FINE-TUNING
    print("\n--- Starting Fine-Tuning Phase to Improve Accuracy ---")
    base_model.trainable = True
    
    # Freeze all layers except the top 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    # Re-compile with a much lower learning rate
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=1e-5),
                  metrics=['accuracy'])
                  
    fine_tune_epochs = min(epochs, 5) # Additional epochs for fine-tuning
    total_epochs = epochs + fine_tune_epochs
    
    model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=validation_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )

    print(f"\nSaving model to {MODEL_NAME}...")
    model.save(MODEL_NAME)
    
    # Save class indices
    import json
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    print("Class indices saved to class_indices.json.")
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FoodLens Model")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    args = parser.parse_args()
    train_model(args.epochs)
