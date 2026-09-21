import Definitions

import os.path as osp
import pandas as pd
import joblib


class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")

        # Define la carpeta donde se encuentran los componentes entrenados que utilizará la aplicación.
        self.models_path = osp.join(Definitions.ROOT_DIR, "resources/models")

        # Construye la ruta del vectorizador TF-IDF entrenado.
        self.vectorizer_path = osp.join(self.models_path, "vectorizador_tfidf.joblib")

        # Construye la ruta del modelo SVD utilizado para reducir la dimensionalidad de la representación TF-IDF.
        self.svd_path = osp.join(self.models_path,"svd.joblib")

        # Construye la ruta del modelo final de clasificación: regresión logística.
        self.model_path = osp.join(self.models_path, "modelo.joblib")

        # Carga los tres componentes entrenados.
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.svd = joblib.load(self.svd_path)
        self.model = joblib.load(self.model_path)

    
    def predict(self, texto):
        """
        Clasifica un texto utilizando los componentes entrenados en el Microproyecto 2.

        Parámetro:
            texto (str): Texto libre ingresado por el usuario.

        Retorna:
            ods_predicho: número del ODS predicho.
            probabilidad_predicha: probabilidad estimada para el ODS predicho.
            probabilidades: DataFrame con las probabilidades de todos los ODS.
        """

        print("ModelController.predict ->")

        # Validamos que se haya recibido un texto.
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("Debe ingresar un texto para realizar la predicción.")

        # El vectorizador aplica el mismo preprocesamiento utilizado
        # durante el entrenamiento y genera la representación TF-IDF.
        X_tfidf = self.vectorizer.transform([texto])

        # Aplicamos la reducción de dimensionalidad SVD entrenada.
        X_reduced = self.svd.transform(X_tfidf)

        # Obtenemos el ODS predicho.
        ods_predicho = self.model.predict(X_reduced)[0]

        # Calculamos la probabilidad estimada para cada ODS.
        probabilidades_modelo = self.model.predict_proba(X_reduced)[0]

        # Relacionamos cada probabilidad con la clase correspondiente.
        probabilidades = pd.DataFrame({"ODS": self.model.classes_, "Probabilidad": probabilidades_modelo})

        # Recuperamos la probabilidad correspondiente al ODS predicho.
        probabilidad_predicha = probabilidades.loc[probabilidades["ODS"] == ods_predicho, "Probabilidad"].iloc[0]

        return ods_predicho, probabilidad_predicha, probabilidades

