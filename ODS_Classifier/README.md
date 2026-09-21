# MAIA-4211_202611_AML

# Clasificador de textos por ODS

Aplicación desarrollada en Streamlit para clasificar textos según los Objetivos de Desarrollo Sostenible (ODS).

## Funcionalidades

La aplicación permite dos formas de entrada de información.

### Ingreso de texto

El usuario puede escribir un texto libre y obtener:

- ODS principal predicho.
- Nombre e imagen del ODS.
- Probabilidad estimada.
- Otros ODS relacionados con probabilidad superior al 10 %.

### Carga de archivo Excel

La aplicación permite cargar archivos `.xlsx` con una columna que contenga los textos a clasificar.

El usuario puede seleccionar una fila de la tabla y obtener la clasificación correspondiente.

Si el archivo contiene una columna `ODS`, la aplicación también muestra:

- La clase real.
- La clase predicha.
- Si la clasificación fue correcta o incorrecta.

## Pipeline de predicción

El proceso utilizado por la aplicación es:

`Texto → Preprocesamiento con spaCy → TF-IDF → Truncated SVD → Regresión Logística → ODS predicho`

## Componentes principales

- `streamlit_app.py`: interfaz de usuario.
- `src/ModelController.py`: carga y ejecución del modelo.
- `src/DataPreprocessing.py`: preprocesamiento de los textos.
- `resources/models/`: componentes entrenados.
- `resources/images/`: imágenes de los ODS.
- `requirements.txt`: dependencias de la aplicación.

## Formato recomendado del archivo Excel

El archivo puede tener una estructura como:

| textos | ODS |
|---|---:|
| Texto relacionado con educación | 4 |
| Texto relacionado con agua potable | 6 |
| Texto relacionado con instituciones | 16 |

La columna `ODS` es opcional. Cuando está disponible, la aplicación la utiliza como clase real para comparar el resultado de la predicción.

## Ejecución local

Desde la carpeta:

`ODS_Classifier/resources/batch`

ejecutar:

```powershell
.\streamlit.bat