import re
import spacy

class DataPreprocessing:
    """
    Clase encargada del preprocesamiento de los textos.
    Reproduce el mismo procedimiento utilizado durante el entrenamiento del modelo del Microproyecto 2.
    """

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        # Se carga el mismo modelo de spaCy utilizado durante el desarrollo y entrenamiento.
        self.nlp = spacy.load("es_core_news_sm")

    def transform(self, texto):
        """
        Limpia y lematiza un texto en español para preparar su análisis.

        Parámetro:
            texto (str): Texto original de un documento.

        Retorna:
            str: Lemas separados por espacios, después de limpiar el texto y excluir tokens no alfabéticos y palabras vacías.
        """

        print("DataPreprocessing.transform ->")

        # Unifica mayúsculas y minúsculas para evitar diferencias de escritura en la representación del vocabulario.
        texto = texto.lower()
        # Eliminar URLs.
        texto = re.sub(r'http\S+|www\S+', ' ', texto)
        # Eliminar correos electrónicos.
        texto = re.sub(r'\S+@\S+', ' ', texto)
        # Reemplaza guiones y rayas por espacios.
        texto = re.sub(r"[-‐-]", " ", texto)
        # Procesa el texto con spaCy.
        doc = self.nlp(texto)

        # Conserva tokens alfabéticos, elimina palabras vacías y utiliza la forma lematizada.
        # También se verifica que el lema obtenido no sea una stopword.
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop
            and token.lemma_ not in self.nlp.Defaults.stop_words]

        # Une los lemas para que el resultado pueda ser utilizado posteriormente por TF-IDF.
        return ' '.join(tokens)
    

