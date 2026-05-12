# Report W&B - Clasificacion de latidos ECG

## Objetivo

El objetivo del experimento es entrenar y evaluar una CNN 1D para clasificar
latidos ECG del dataset MIT-BIH en cinco clases. La metrica principal elegida
es F1 macro porque el conjunto esta bastante desbalanceado: la clase normal es
mayoritaria y algunas clases patologicas tienen muchos menos ejemplos.

## Configuracion del experimento

- Dataset: `shayanfazeli/heartbeat`, particion MIT-BIH Arrhythmia.
- Entrada del modelo: latido ECG preprocesado con 187 puntos temporales.
- Modelo: CNN 1D con 3 bloques convolucionales, BatchNorm, ReLU, MaxPool y
  Dropout.
- Semilla: 42.
- Epocas: 6.
- Learning rate: 0.001.
- Batch size: 256.
- Validacion: 15% del conjunto de entrenamiento, con estratificacion.

## Resultados

| Metrica | Valor |
| --- | ---: |
| Mejor F1 macro en validacion | 0.8932 |
| Loss test | 0.0785 |
| Accuracy test | 0.9784 |
| F1 macro test | 0.8766 |
| F1 weighted test | 0.9772 |

## Analisis

La accuracy es alta, pero por si sola no seria suficiente para elegir el
modelo, porque el dataset esta desbalanceado. Por eso se usa F1 macro como
criterio principal. El modelo llega a un F1 macro de test de 0.8766, lo que
indica que no solo acierta la clase normal, sino que tambien aprende patrones
de las clases minoritarias.

La curva de entrenamiento mejora de forma estable durante las 6 epocas. El F1
macro de validacion sube desde 0.6561 hasta 0.8932, sin una divergencia clara
entre entrenamiento y validacion. Esto sugiere que el modelo aprende patrones
utiles de la morfologia del latido sin un sobreajuste fuerte en esta ejecucion.

## Artefacto

El modelo final se guarda como artefacto versionable en `models/ecg_cnn.pt`.
En W&B se registra tambien `models/metadata.json`, que contiene las metricas
finales y las clases del problema. Esto permite recuperar el modelo usado por
la API y relacionarlo con el run que lo produjo.

## Componentes W&B usados

- Configuracion del run: hiperparametros, rutas y datos del proyecto.
- Metricas por epoca: loss, accuracy y F1 macro de entrenamiento/validacion.
- Metricas finales de test.
- Artefacto del modelo entrenado.
- Report con el analisis del experimento.

