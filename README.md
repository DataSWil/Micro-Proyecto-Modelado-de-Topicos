# MAIA-4211_202611_MLNS_Deploy
Plantilla para carga de la información de MAIA

# Microproyecto 2 - Machine Learning No Supervisado

## Descripción

Este proyecto desarrolla un proceso de análisis y clasificación de textos relacionados con los Objetivos de Desarrollo Sostenible (ODS).

El desarrollo incluye la preparación de los textos, su representación mediante TF-IDF, la reducción de dimensionalidad con Truncated SVD y la evaluación de modelos de clasificación.

Como parte del proyecto se desarrolló una aplicación web en Streamlit que permite utilizar el modelo final con nuevos textos.

## Aplicación de clasificación de ODS

La aplicación se encuentra en la carpeta:

`ODS_Classifier/`

Permite:

- Ingresar un texto manualmente.
- Cargar un archivo Excel `.xlsx` con varios textos.
- Seleccionar un texto desde el archivo.
- Obtener el ODS principal predicho.
- Mostrar la probabilidad estimada de la predicción.
- Mostrar otros ODS relacionados con probabilidad superior al 10 %.
- Comparar la predicción con la clase real cuando el archivo contiene
  una columna `ODS`.

## Pipeline del modelo

El proceso utilizado para realizar las predicciones es:

`Texto → Preprocesamiento con spaCy → TF-IDF → Truncated SVD → Regresión Logística → ODS predicho`

## Estructura principal

- `ODS_Classifier/`: aplicación web desarrollada en Streamlit.
- `ODS_Classifier/src/`: lógica de preprocesamiento y predicción.
- `ODS_Classifier/resources/models/`: componentes entrenados del modelo.
- `ODS_Classifier/resources/images/`: imágenes utilizadas en la interfaz.
- `ODS_Classifier/requirements.txt`: dependencias necesarias para ejecutar la aplicación.