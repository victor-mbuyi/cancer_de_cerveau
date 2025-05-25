import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import os
import argparse

# Define transfer learning model
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

# Define paths
train_dir = '/Users/mac/Desktop/CV A1/breast/projetFini/compilationFini/codetest/data/brain_tumor/training'
test_dir = '/Users/mac/Desktop/CV A1/breast/projetFini/compilationFini/codetest/data/brain_tumor/testing'

# Validate paths
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory {train_dir} does not exist.")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Testing directory {test_dir} does not exist.")

# Load dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Brain Tumor CNN with Transfer Learning')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()

    model = create_model()
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=test_generator
    )
    model.save('Victor_model.tensorflow')
    print("TensorFlow model saved as Victor_model.tensorflow")