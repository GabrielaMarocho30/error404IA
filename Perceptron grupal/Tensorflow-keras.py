import tensorflow as tf
import numpy as np

# Datos de entrada (combinaciones binarias para una puerta AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
D = np.array([0, 0, 0, 1], dtype=np.float32)

# Construcción del modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=2) #No se define funcion de activacion
])

# Compilación del modelo
modelo.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Entrenamiento del modelo
modelo.fit(X, D, epochs=500, verbose=1)

# Evaluación del modelo
predicciones = modelo.predict(X)
print("Predicciones del modelo:")
print(np.round(predicciones)) # Redondeamos las predicciones para obtener valores binarios (0 o 1)
