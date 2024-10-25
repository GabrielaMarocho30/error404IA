# APLICACIÓN DE MACHINE LEARNING PARA PREDICCIONES DE FUTBOL
El presente proyecto tiene como objetivo analizar y aplicar Random Forest para predecir el rendimiento de los equipos de premier league de inglaterra, temporada 2023/2024 a partir del Dataset. de manera que con la información obtenida se puedan realizar predicciones para apuestas de esta liga. Para ello se aplico la metodología SEMA.
## Contenido
- Requisitos
- Instalación
- Implementación de SEMMA
- Ejecución
- Resultados
- Mejoras Futuras

## Requisitos
Python 3.7 o superior
## Instalación
1. Clona el repositorio
2. Instala las dependencia
3. Asegúrate de tener el dataset en el directorio del proyecto o en la ruta especificada en el código.


## Implementación de SEMMA
1. Sample (Muestra)
Cargamos el conjunto de datos y revisamos su equilibrio.
Revisamos la distribución de victorias locales y visitantes para analizar la necesidad de balanceo.
2. Explore (Explorar)
Visualizamos distribuciones de variables clave, como goles y posesión (%), para comprender patrones y distribuciones.
Realizamos un análisis de correlación para identificar variables predictivas.
3. Modify (Modificar)
Seleccionamos y estandarizamos características relevantes (e.g., Goles, Tiros, Tarjetas).
Preparamos la variable objetivo para clasificar los equipos en función de victorias locales vs. visitantes.
4. Model (Modelar)
Entrenamos un modelo de Random Forest para clasificar victorias con los parámetros:

- ```n_estimators=100```
- ``` random_state=42```
  Dividimos los datos en entrenamiento (70%) y prueba (30%).
5. Assess (Evaluar)
  Evaluamos el modelo utilizando precisión, matriz de confusión y reporte de clasificación.
  Analizamos el rendimiento y sesgo del modelo, sugiriendo posibles optimizaciones.

## Resultados
- Precisión del modelo: 66.67% en datos de prueba.
- Matriz de confusión: muestra la distribución de predicciones correctas e incorrectas para victorias locales y visitantes.
- Reporte de Clasificación: precisión y recall desglosados por clase.
## Mejoras Futuras
- Balanceo de Clases: Implementar técnicas de balanceo para mejorar la clasificación de equipos visitantes.
- Ajuste de Hiperparámetros: Realizar una búsqueda de hiperparámetros para mejorar el rendimiento del modelo.
- Más Características: Incluir variables adicionales que mejoren la predicción del modelo.
