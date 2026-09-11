
'''
import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime


def capture_samples(path, margin_frame=7, min_cant_frames=5, delay_frames=10):
    
    ### CAPTURA DE MUESTRAS PARA UNA PALABRA
    Recibe como parámetro la ubicación de guardado y guarda los frames
    
    `path` ruta de la carpeta de la palabra \n
    `margin_frame` cantidad de frames que se ignoran al comienzo y al final \n
    `min_cant_frames` cantidad de frames minimos para cada muestra \n
    `delay_frames` cantidad de frames que espera antes de detener la captura después de no detectar manos
    
    create_folder(path)
    
    count_frame = 0
    frames = []
    fix_frames = 0
    recording = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) >= min_cant_frames + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    frames = frames[: - (margin_frame + delay_frames)]
                    today = datetime.now().strftime('%y%m%d%H%M%S%f')
                    output_folder = os.path.join(path, f"sample_{today}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                
                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0,220, 100))
            
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "adios"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
---------
import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime


def capture_samples(path, margin_frame=7, min_cant_frames=5, delay_frames=10):
    
    ### CAPTURA DE MUESTRAS PARA UNA PALABRA
    Recibe como parámetro la ubicación de guardado y guarda los frames
    
    path ruta de la carpeta de la palabra \n
    margin_frame cantidad de frames que se ignoran al comienzo y al final \n
    min_cant_frames cantidad de frames mínimos para cada muestra \n
    delay_frames cantidad de frames que espera antes de detener la captura después de no detectar manos
    
    create_folder(path)  # Asegura que la carpeta de destino se cree
    
    frames = []
    capturing = False  # Controla si la captura está activa
    stop_capturing = False  # Controla si la captura está activa
    save_index = 0
    text_hint = ""
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            # Detectar si se presiona la tecla 'c' para iniciar o detener la captura
            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'):
                capturing = True
                stop_capturing = False
            elif key == ord('x'):
                stop_capturing = True
                capturing = False

            if capturing:
                cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                frames.append(np.asarray(frame))
            elif stop_capturing:
                # Guardar los frames capturados
                today = datetime.now().strftime('%y%m%d%H%M%S%f')
                output_folder = os.path.join(path, f"sample_{today}")
                create_folder(output_folder)
                print(f"Guardando {len(frames)} frames en {output_folder}")  # Verifica que llegue a esta parte
                save_frames(frames, output_folder) 
                frames = [] 
                stop_capturing = False   
                save_index = save_index + 1
            else:
                text_hint = '[{}] | "C" to Capture / "X" to stop'.format(save_index)
                cv2.putText(image, text_hint, FONT_POS, FONT, FONT_SIZE, (100, 0, 255))

            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)

            # Detener el programa con 'q'
            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    word_name = "bien"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)'''

import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime


def capture_samples(path, margin_frame=7, min_cant_frames=5, delay_frames=10):
    '''
    ### CAPTURA DE MUESTRAS PARA UNA PALABRA
    Recibe como parámetro la ubicación de guardado y guarda los frames
    
    path ruta de la carpeta de la palabra \n
    margin_frame cantidad de frames que se ignoran al comienzo y al final \n
    min_cant_frames cantidad de frames mínimos para cada muestra \n
    delay_frames cantidad de frames que espera antes de detener la captura después de no detectar manos
    '''
    create_folder(path)  # Asegura que la carpeta de destino se cree
    
    frames = []
    capturing = False  # Controla si la captura está activa
    stop_capturing = False  # Controla si la captura está activa
    save_index = 0
    frame_index = 1
    text_hint = ""
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            # Detectar si se presiona la tecla 'c' para iniciar o detener la captura
            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'):
                capturing = True
                stop_capturing = False
            elif key == ord('x'):
                stop_capturing = True
                capturing = False

            if capturing:
                text_hint = 'Capturando... | [{}] frames'.format(frame_index)
                cv2.putText(image, text_hint, FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                frames.append(np.asarray(frame))
                frame_index = frame_index + 1
            elif stop_capturing:
                # Guardar los frames capturados
                today = datetime.now().strftime('%y%m%d%H%M%S%f')
                output_folder = os.path.join(path, f"sample_{today}")
                create_folder(output_folder)
                print(f"Guardando {len(frames)} frames en {output_folder}")  # Verifica que llegue a esta parte
                save_frames(frames, output_folder) 
                frames = [] 
                stop_capturing = False   
                save_index = save_index + 1
                frame_index = 1
            else:
                text_hint = '[{}] | "C" to Capture / "X" to stop'.format(save_index)
                cv2.putText(image, text_hint, FONT_POS, FONT, FONT_SIZE, (100, 0, 255))

            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)

            # Detener el programa con 'q'
            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    word_name = "porfavor"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)