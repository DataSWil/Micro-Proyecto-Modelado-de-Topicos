#  We ensure proper path handling in Python
import Definitions
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os.path as osp

from src.ModelController import ModelController

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ---------------------------------------------------------

st.set_page_config(
    layout="centered", page_title="Clasificador de ODS", page_icon="🌍")

# Fondo blanco para mantener una presentación uniforme.
st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
            color: #111111;
        }

        .stApp p,
        .stApp label,
        .stApp h1,
        .stApp h2,
        .stApp h3 {
            color: #111111;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ESTADO DE LA APLICACIÓN
# =========================================================

# Este contador permite limpiar los campos de texto y archivos
# cuando el usuario desea realizar una nueva consulta.
if "consulta_id" not in st.session_state:
    st.session_state["consulta_id"] = 0

# Aquí se almacena temporalmente el resultado de la predicción
# para conservarlo mientras Streamlit actualiza la página.
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None


# ---------------------------------------------------------
# INFORMACIÓN VISUAL DE LOS ODS
# ---------------------------------------------------------

ODS_INFO = {1: {"nombre": "Fin de la pobreza", "imagen": "ods_1.png"}, 
            2: {"nombre": "Hambre cero","imagen": "ods_2.png"},
            3: {"nombre": "Salud y bienestar", "imagen": "ods_3.png"},
            4: {"nombre": "Educación de calidad", "imagen": "ods_4.png"},
            5: {"nombre": "Igualdad de género", "imagen": "ods_5.png"},
            6: {"nombre": "Agua limpia y saneamiento", "imagen": "ods_6.png"},
            7: {"nombre": "Energía asequible y no contaminante", "imagen": "ods_7.png"},
            8: {"nombre": "Trabajo decente y crecimiento económico", "imagen": "ods_8.png"},
            9: {"nombre": "Industria, innovación e infraestructura", "imagen": "ods_9.png"},
            10: {"nombre": "Reducción de las desigualdades", "imagen": "ods_10.png"},
            11: {"nombre": "Ciudades y comunidades sostenibles", "imagen": "ods_11.png"},
            12: {"nombre": "Producción y consumo responsables", "imagen": "ods_12.png"},
            13: {"nombre": "Acción por el clima", "imagen": "ods_13.png"},
            14: {"nombre": "Vida submarina", "imagen": "ods_14.png"},
            15: {"nombre": "Vida de ecosistemas terrestres", "imagen": "ods_15.png"},
            16: {"nombre": "Paz, justicia e instituciones sólidas", "imagen": "ods_16.png"}}


# Carpeta donde se almacenan las imágenes utilizadas.
IMAGES_PATH = osp.join(Definitions.ROOT_DIR, "resources", "images")

# ---------------------------------------------------------
# CARGA DEL MODELO
# ---------------------------------------------------------

# Streamlit ejecuta nuevamente el archivo cuando ocurre una interacción. 
# cache_resource evita cargar los tres modelos desde disco en cada ejecución.
@st.cache_resource
def cargar_controlador():
    return ModelController()


ctrl = cargar_controlador()


# ---------------------------------------------------------
# FUNCIONES DE LA INTERFAZ
# ---------------------------------------------------------

def realizar_prediccion(texto):
    """
    Envía el texto al ModelController y almacena el resultado.
    """
    try:
        resultado = ctrl.predict(texto)
        st.session_state["resultado"] = resultado

    except ValueError as error:
        st.warning(str(error))

    except Exception as error:
        st.error(f"Ocurrió un error al realizar la predicción: {error}")


def nueva_consulta():
    """
    Limpia la consulta actual, pero conserva la aplicación disponible para realizar otra clasificación.
    """
    st.session_state["resultado"] = None
    # Al cambiar este número se generan nuevamente los campos de entrada vacíos.
    st.session_state["consulta_id"] += 1
    st.rerun()


def finalizar():
    """
    Limpia completamente el estado de la aplicación y regresa a la pantalla inicial.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()


def mostrar_resultado():
    """
    Presenta el resultado principal y las demás relaciones con ODS cuya probabilidad sea superior al 10 %.
    """

    resultado = st.session_state["resultado"]

    # Si todavía no existe una predicción, no se muestra nada.
    if resultado is None:
        return

    ods_predicho, probabilidad_predicha, probabilidades = resultado


    # Convertimos a int porque las clases del modelo pueden venir almacenadas como enteros de NumPy.
    ods_predicho = int(ods_predicho)

    informacion = ODS_INFO.get(ods_predicho)

    if informacion is None:
        st.error(f"No se encontró información visual para el ODS {ods_predicho}.")
        return

    
    # RESULTADO PRINCIPAL
    st.divider()
    st.subheader("Resultado de la clasificación")

    ruta_imagen = osp.join(IMAGES_PATH, informacion["imagen"])
    col_imagen, col_resultado = st.columns([1, 2])

    with col_imagen:
        # Se muestra el icono correspondiente al ODS predicho.
        if osp.exists(ruta_imagen):
            st.image(ruta_imagen, width=180)
        else:
            st.warning(
                f"No se encontró la imagen {informacion['imagen']}.")
            
    with col_resultado:
        st.markdown(
            f"### ODS {ods_predicho}")
        st.markdown(
            f"**{informacion['nombre']}**")

        st.metric(label="Probabilidad estimada", value=f"{probabilidad_predicha * 100:.2f} %")

    
    # OTRAS ODS RELACIONADAS
    st.subheader("Otras ODS relacionadas:")

    # Se excluye el ODS principal porque ya fue presentado arriba.
    # Solo se conservan las demás probabilidades superiores al 10 %.
    otras_ods = probabilidades[(probabilidades["ODS"] != ods_predicho) & 
                               (probabilidades["Probabilidad"] > 0.10)].copy()

    if otras_ods.empty:
        st.info("No se encontraron otras ODS con una probabilidad superior al 10 %.")
    else:
        # Incorporamos el nombre descriptivo de cada ODS.
        otras_ods["Nombre"] = otras_ods["ODS"].apply(
            lambda ods: (
                f"ODS {int(ods)} - "
                f"{ODS_INFO.get(int(ods), {}).get('nombre', '')}"))

        # Convertimos las probabilidades a porcentaje.
        otras_ods["Probabilidad (%)"] = (otras_ods["Probabilidad"] * 100)

        # Se ordenan de menor a mayor para que en el gráfico la probabilidad más alta quede visualmente arriba.
        otras_ods = otras_ods.sort_values(by="Probabilidad (%)", ascending=True)


        
        # GRÁFICA HORIZONTAL
        
        # Creamos el gráfico horizontal.
        fig, ax = plt.subplots(figsize=(8, max(2.5, len(otras_ods) * 0.8)))
        barras = ax.barh( otras_ods["Nombre"], otras_ods["Probabilidad (%)"])
        ax.set_xlabel("Probabilidad (%)")
        ax.set_ylabel("")

        # Agregamos el porcentaje al final de cada barra.
        for barra, valor in zip(barras, otras_ods["Probabilidad (%)"]):
            ax.text(barra.get_width() + 0.3, barra.get_y() + barra.get_height() / 2, 
                    f"{valor:.2f} %", va="center")
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        # Liberamos la figura después de mostrarla.
        plt.close(fig)

    # -----------------------------------------------------
    # BOTONES POSTERIORES AL RESULTADO
    # -----------------------------------------------------

    st.divider()

    col_nueva, col_finalizar = st.columns(2)

    with col_nueva:
        if st.button("🔄 Nueva consulta", use_container_width=True):
            nueva_consulta()

    with col_finalizar:
        if st.button("Finalizar", use_container_width=True):
            finalizar()


# ---------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------

ruta_encabezado = osp.join(IMAGES_PATH, "encabezado_onu.png")

if osp.exists(ruta_encabezado):
    st.image(ruta_encabezado, use_container_width=True)
st.title("Clasificación de textos por ODS")
st.write("Ingrese un texto o seleccione un documento desde un archivo "
    "para identificar el Objetivo de Desarrollo Sostenible "
    "más relacionado.")


# ---------------------------------------------------------
# SELECCIÓN DEL TIPO DE ENTRADA
# ---------------------------------------------------------

tipo_entrada = st.radio("¿Cómo desea ingresar la información?", 
                        ["Ingresar texto", "Adjuntar archivo de textos"], 
                        key=f"tipo_entrada_{st.session_state['consulta_id']}")

## =========================================================
# OPCIÓN 1: INGRESAR TEXTO MANUALMENTE
# =========================================================

if tipo_entrada == "Ingresar texto":
    texto = st.text_area("Ingrese el texto que desea clasificar:", height=200, 
                         placeholder="Escriba aquí el texto...", key=(f"texto_"
                                                                      f"{st.session_state['consulta_id']}"))

    if st.button("Enviar", key=(f"enviar_texto_" 
                                f"{st.session_state['consulta_id']}")):

        realizar_prediccion(texto)


# =========================================================
# OPCIÓN 2: ADJUNTAR ARCHIVO
# =========================================================

else:

    uploaded_file = st.file_uploader("Seleccione un archivo CSV que contenga los textos:", 
                                     type="csv", accept_multiple_files=False, 
                                     key=(f"archivo_" 
                                          f"{st.session_state['consulta_id']}"))

    if uploaded_file is not None:
        try:
            # Cargamos el archivo en un DataFrame.
            input_df = pd.read_csv(uploaded_file)

            if input_df.empty:
                st.warning("El archivo no contiene registros.")

            else:
                st.subheader("Textos disponibles")

                # Intentamos identificar automáticamente una columna que contenga textos.
                posibles_columnas = [columna for columna in input_df.columns if columna.lower() 
                                     in ["texto", "text","textos", "texto_original"]]

                if posibles_columnas:
                    columna_texto = posibles_columnas[0]
                    st.caption(
                        f"Columna de texto detectada: " 
                        f"`{columna_texto}`")

                else:
                    # Si no reconocemos automáticamente la columna, el usuario puede seleccionarla.
                    columna_texto = st.selectbox("Seleccione la columna que contiene los textos:", 
                                                 input_df.columns)

                # Mostramos la tabla de selección de una fila.
                event = st.dataframe(input_df, on_select="rerun", selection_mode="single-row", 
                                     use_container_width=True, hide_index=True, 
                                     key=(f"tabla_" 
                                          f"{st.session_state['consulta_id']}"))

                st.caption("Seleccione una fila para analizar su texto.")

                # Se recupera la fila elegida por el usuario.
                if event.selection.rows:
                    indice = event.selection.rows[0]
                    texto_seleccionado = input_df.iloc[indice][columna_texto]

                    if (pd.isna(texto_seleccionado) or not str(texto_seleccionado).strip()):
                        st.warning("La fila seleccionada no contiene "
                                   "un texto válido.")

                    else:
                        st.markdown("**Texto seleccionado:**")
                        st.write(str(texto_seleccionado))

                        realizar_prediccion(str(texto_seleccionado))

        except Exception as error:
            st.error(f"No fue posible leer el archivo: {error}")


# ---------------------------------------------------------
# FINALIZAR
# ---------------------------------------------------------

mostrar_resultado()