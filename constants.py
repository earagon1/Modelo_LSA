import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 126
MODEL_FRAMES = 15

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# Las claves son los word_id: deben coincidir con models/words.json y con los
# .h5 de data/keypoints. Los valores son solo el texto que se muestra y se
# pronuncia, asi que se pueden editar sin reentrenar.
words_text = {
    "hola": "Hola",
    "adios": "Adiós",
    "como_estas": "¿Cómo estás?",
    "bien": "Bien",
    "chau": "Chau",
    "gracias": "Gracias",
    "e":"E",
    "v":"V",
    "l":"L",
    "i":"I",
    "n":"N",
    "mi_nombre_es": "Mi nombre es",
    "por_favor": "Por favor"
}
