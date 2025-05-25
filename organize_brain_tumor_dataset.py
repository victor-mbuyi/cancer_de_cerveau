import os
import shutil

# Paths
data_dir = 'data/brain_tumor'
train_dir = '/Users/mac/Desktop/CV A1/breast/projetFini/compilationFini/codetest/data/brain_tumor/training'
test_dir = '/Users/mac/Desktop/CV A1/breast/projetFini/compilationFini/codetest/data/brain_tumor/testing'

# Classes
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create class subdirectories
for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

# Copy Training folder images to train_dir
for class_name in classes:
    train_class_path = os.path.join(data_dir, 'Training', class_name)
    if os.path.exists(train_class_path):
        for img in os.listdir(train_class_path):
            if img.endswith(('.jpg', '.png')):
                shutil.copy(
                    os.path.join(train_class_path, img),
                    os.path.join(train_dir, class_name, img)
                )

# Copy Testing folder images to test_dir
for class_name in classes:
    test_class_path = os.path.join(data_dir, 'Testing', class_name)
    if os.path.exists(test_class_path):
        for img in os.listdir(test_class_path):
            if img.endswith(('.jpg', '.png')):
                shutil.copy(
                    os.path.join(test_class_path, img),
                    os.path.join(test_dir, class_name, img)
                )

print("Dataset organized into data/train and data/test.")