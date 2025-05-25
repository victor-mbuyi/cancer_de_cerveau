import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, session
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, f1_score
import base64
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Define paths
BASE_DIR = '/Users/mac/Desktop/CV A1/breast/projetFini/compilationFini/codetest'
TEST_DIR = os.path.join(BASE_DIR, 'data/brain_tumor/testing')
MODEL_DIR = BASE_DIR
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, 'Victor_model.torch')
TF_MODEL_PATH = os.path.join(MODEL_DIR, 'Victor_model.tensorflow')

# Classes
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define PyTorch model
class BrainTumorResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorResNet, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Load PyTorch model
pytorch_model = BrainTumorResNet()
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu')))
pytorch_model.eval()

# Load TensorFlow model
tf_model = load_model(TF_MODEL_PATH)

# Define transforms for PyTorch
pytorch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define preprocessing for TensorFlow
def preprocess_tf_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Compute metrics on test dataset
def compute_metrics(model_type):
    true_labels = []
    pred_labels = []
    if model_type == 'pytorch':
        test_dataset = torchvision.datasets.ImageFolder(TEST_DIR, transform=pytorch_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        model = pytorch_model
        device = torch.device('cpu')
        model.to(device)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())
    else:  # tensorflow
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        predictions = tf_model.predict(test_generator)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = test_generator.classes

    accuracy = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    return {'accuracy': accuracy, 'recall': recall, 'f1': f1}

# Precomputed metrics
pytorch_metrics = compute_metrics('pytorch')
tf_metrics = compute_metrics('tensorflow')

# Classify image and encode image for display
def classify_image(image, model_type):
    if model_type == 'pytorch':
        image_tensor = pytorch_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = pytorch_model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            return CLASSES[predicted.item()]
    else:  # tensorflow
        image_array = preprocess_tf_image(image)
        prediction = tf_model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return CLASSES[predicted_class]

# Encode image to base64 for display
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_data = session.get('image_data')  # Retrieve image_data from session

    if request.method == 'POST':
        model_type = request.form['model']
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert('RGB')
            prediction = classify_image(image, model_type)
            image_data = encode_image(image)
            session['image_data'] = image_data  # Store image_data in session

    return render_template('index.html', prediction=prediction, image_data=image_data,
                           pytorch_metrics=pytorch_metrics, tf_metrics=tf_metrics)

if __name__ == '__main__':
    app.run(debug=True)