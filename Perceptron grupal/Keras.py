import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Paso 1: Inicializar los datos de entrada (X) y las salidas deseadas (D)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)  # Entradas
D = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Salidas deseadas (puerta lógica AND)

# Paso 2: Definir el modelo del perceptrón usando Keras
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))  # 2 entradas, 1 salida con activación sigmoide

# Paso 3: Compilar el modelo
# Utilizamos descenso de gradiente estocástico (SGD) como optimizador y entropía cruzada binaria como función de pérdida
optimizer = SGD(learning_rate=0.1)  # Coeficiente de aprendizaje
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Paso 4: Entrenar el modelo
model.fit(X, D, epochs=1000, verbose=1)  # Entrenamiento con 1000 épocas

# Paso 5: Evaluar el modelo y mostrar los pesos entrenados y el sesgo
weights, bias = model.layers[0].get_weights()
print("\nPesos entrenados:", weights)
print("Sesgo entrenado:", bias)

# Predicciones
print("\nPredicciones:")
predicciones = model.predict(X)
print(np.round(predicciones))  # Redondear las predicciones a 0 o 1 para la puerta AND
