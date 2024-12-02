import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm  # Importamos tqdm para la barra de progreso

# Verificar la versión de PyTorch y si CUDA está disponible
print(torch.__version__)  # Versión de PyTorch
print(torch.cuda.is_available())  # Si CUDA está disponible
print(torch.cuda.current_device())  # ID de la GPU actual
print(torch.cuda.get_device_name(0))  # Nombre de la primera GPU


# Verificar si hay GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_per_process_memory_fraction(0.8, device=0)  # 80% de la memoria de la GPU
print(f'Usando dispositivo: {device}')
torch.backends.cudnn.benchmark = True  # Habilitar la optimización para el hardware


# Directorio principal de las imágenes
dataset_dir = "D:/EFRAM/archive"

# Rutas de entrenamiento, validación y pruebas
datos_entrenamiento = dataset_dir + "/Train"
datos_validacion = dataset_dir + "/Validation"
datos_pruebas = dataset_dir + "/Test"

# Transformaciones para el preprocesamiento de imágenes
train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar los datasets
train_dataset = datasets.ImageFolder(root=datos_entrenamiento, transform=train_transform)
val_dataset = datasets.ImageFolder(root=datos_validacion, transform=val_transform)
test_dataset = datasets.ImageFolder(root=datos_pruebas, transform=val_transform)

# DataLoader para cargar los datos
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Definición del modelo de red neuronal convolucional (CNN)
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 32, 128)  # Ajustar según la imagen de entrada
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # 2 clases

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Aplanar la imagen
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Instanciar el modelo
model = CNN_Model().to(device)

# Definir el optimizador y la función de pérdida
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Entrenamiento del modelo con barra de progreso
epochs = 25
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Barra de progreso para el entrenamiento
    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Actualizar la barra de progreso
            tepoch.set_postfix(loss=running_loss / len(train_loader), accuracy=100 * correct / total)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validación
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'modelo_clasificacion.pth')
print("Modelo guardado como 'modelo_clasificacion.pth'")

# Resumen del modelo (opcional)
summary(model, input_size=(3, 256, 256))
