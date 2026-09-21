import re
import spacy

class DataPreprocessing:

    def __init__(self):
        """
        Carga el modelo de spaCy utilizado durante el entrenamiento del proyecto.
        """

        self.nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])


    def transform(self, texto):
        """
        Aplica exactamente el mismo preprocesamiento utilizado durante el entrenamiento.
        """

        # Convertimos el texto a minúsculas.
        texto = texto.lower()

        # Eliminamos direcciones web.
        texto = re.sub(r"http\S+|www\S+", " ", texto)

        # Eliminamos direcciones de correo electrónico.
        texto = re.sub(r"\S+@\S+", " ", texto)

        # Sustituimos guiones por espacios.
        texto = re.sub(r"[-‐-]", " ", texto)

        # Procesamos el texto con spaCy.
        doc = self.nlp(texto)

        # Conservamos palabras alfabéticas, eliminamos stopwords y utilizamos el lema.
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop 
                  and token.lemma_ not in self.nlp.Defaults.stop_words]

        return " ".join(tokens)