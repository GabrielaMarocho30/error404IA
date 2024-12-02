from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Configuración de Flask
app = Flask(__name__)

# Definir la arquitectura del modelo (igual al entrenado)
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Cargar el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_Model().to(device)
model.load_state_dict(torch.load('modelo_clasificacion.pth', map_location=device))
model.eval()

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Rutas de la aplicación
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen subida por el usuario
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No se proporcionó ninguna imagen'})

    # Abrir y preprocesar la imagen
    image = Image.open(file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Realizar predicción
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()

    # Mapear las probabilidades a las clases
    class_names = ['Fake', 'Real']
    response = {
        'predictions': [
            {'class': class_names[i], 'probability': round(prob * 100, 2)}
            for i, prob in enumerate(probabilities)
        ]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)