import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Definir la arquitectura del modelo (debe ser igual al modelo que entrenaste)
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Asegúrate de usar las mismas dimensiones
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # Ajusta el número de clases

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Aplanar la imagen
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Cargar el modelo guardado
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_Model().to(device)
model.load_state_dict(torch.load('modelo_clasificacion.pth', map_location=device))
model.eval()

# Transformaciones para la imagen de prueba
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar y preprocesar una imagen
ruta_imagen = "2.png"  # Cambia a la ruta de tu imagen
imagen = Image.open(ruta_imagen).convert('RGB')
input_tensor = transform(imagen).unsqueeze(0).to(device)

# Realizar predicción con probabilidades
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)  # Convertir logits en probabilidades
    predicted_class = torch.argmax(probabilities, dim=1)

# Mapear la clase predicha a su etiqueta
class_names = ['FAKE', 'REAL']  # Cambia a las etiquetas reales de tus clases
print(f"Predicción: {class_names[predicted_class.item()]}")
print(f"Probabilidades: {probabilities.squeeze().tolist()}")
