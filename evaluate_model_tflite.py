import os
import sys
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import *
from constants import *
from text_to_speech import text_to_speech

# Importaciones para TFLite
import tensorflow as tf

def interpolate_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    
    return interpolated_keypoints

def normalize_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolate_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints

class ModelPredictor:
    """Clase para manejar predicciones tanto con modelos Keras como TFLite"""
    
    def __init__(self, model_path, use_tflite=False):
        self.use_tflite = use_tflite
        self.model_path = model_path
        
        if use_tflite:
            # Cargar modelo TFLite
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Obtener detalles de entrada y salida
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ Modelo TFLite cargado: {model_path}")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Input dtype: {self.input_details[0]['dtype']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
        else:
            # Cargar modelo Keras
            from keras.models import load_model
            self.model = load_model(model_path)
            print(f"✅ Modelo Keras cargado: {model_path}")
    
    def predict(self, input_data):
        """Realiza predicción con el modelo cargado"""
        if self.use_tflite:
            # Asegurar que el input tenga el tipo de dato correcto
            input_dtype = self.input_details[0]['dtype']
            if input_dtype == np.float32:
                input_data = input_data.astype(np.float32)
            
            # Establecer el tensor de entrada
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Ejecutar inferencia
            self.interpreter.invoke()
            
            # Obtener el resultado
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            return output_data[0]  # Retornar primera predicción
        else:
            # Predicción con modelo Keras
            return self.model.predict(input_data)[0]
   
def evaluate_model(src=None, threshold=0.5, margin_frame=1, delay_frames=3, 
                  model_path=None, use_tflite=False, compare_mode=False):
    """
    Evalúa el modelo de reconocimiento de señas.
    
    Args:
        src: Fuente de video (None para webcam, path para archivo)
        threshold: Umbral de confianza para aceptar predicción
        margin_frame: Frames de margen
        delay_frames: Frames de retraso
        model_path: Ruta al modelo (.keras para Keras, .tflite para TFLite)
        use_tflite: Si True, usa modelo TFLite; si False, usa modelo Keras
        compare_mode: Si True, compara ambos modelos simultáneamente
    """
    
    # Si no se especifica modelo_path, usar el de constants
    if model_path is None:
        model_path = MODEL_PATH
    
    kp_seq, sentence = [], []
    word_ids = get_word_ids(WORDS_JSON_PATH)
    
    # Crear predictor principal
    predictor = ModelPredictor(model_path, use_tflite)
    
    # Definir el tipo de modelo desde el inicio
    model_type = "TFLite" if use_tflite else "Keras"
    
    # Si está en modo comparación, cargar ambos modelos
    predictor_compare = None
    if compare_mode:
        # Determinar la ruta del otro modelo
        if use_tflite:
            # Si estamos usando TFLite, cargar también Keras
            keras_path = model_path.replace('.tflite', '.keras')
            if os.path.exists(keras_path):
                predictor_compare = ModelPredictor(keras_path, use_tflite=False)
                print("🔄 Modo comparación: TFLite vs Keras")
        else:
            # Si estamos usando Keras, cargar también TFLite
            tflite_path = model_path.replace('.keras', '.tflite')
            if os.path.exists(tflite_path):
                predictor_compare = ModelPredictor(tflite_path, use_tflite=True)
                print("🔄 Modo comparación: Keras vs TFLite")
    
    count_frame = 0
    fix_frames = 0
    recording = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break

            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(results)
                    kp_seq.append(kp_frame)
            
            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    
                    kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                    kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    
                    # Preparar input para predicción
                    input_data = np.expand_dims(kp_normalized, axis=0)
                    
                    # Hacer predicción principal
                    res = predictor.predict(input_data)
                    predicted_idx = np.argmax(res)
                    confidence = res[predicted_idx] * 100
                    
                    print(f"\n{'='*50}")
                    print(f"[{model_type}] Predicción: {predicted_idx} - {word_ids[predicted_idx]}")
                    print(f"[{model_type}] Confianza: {confidence:.2f}%")
                    
                    # Si está en modo comparación
                    if predictor_compare is not None:
                        res_compare = predictor_compare.predict(input_data)
                        idx_compare = np.argmax(res_compare)
                        conf_compare = res_compare[idx_compare] * 100
                        compare_type = "Keras" if use_tflite else "TFLite"
                        
                        print(f"[{compare_type}] Predicción: {idx_compare} - {word_ids[idx_compare]}")
                        print(f"[{compare_type}] Confianza: {conf_compare:.2f}%")
                        
                        # Análisis de diferencias
                        if predicted_idx != idx_compare:
                            print("⚠️  PREDICCIONES DIFERENTES!")
                            print(f"   {model_type}: {word_ids[predicted_idx]} ({confidence:.2f}%)")
                            print(f"   {compare_type}: {word_ids[idx_compare]} ({conf_compare:.2f}%)")
                        
                        # Diferencias en probabilidades
                        diff = np.abs(res - res_compare)
                        max_diff = np.max(diff)
                        avg_diff = np.mean(diff)
                        
                        print(f"\n📊 Análisis de diferencias:")
                        print(f"   Máxima: {max_diff*100:.4f}%")
                        print(f"   Promedio: {avg_diff*100:.4f}%")
                        
                        # Mostrar top 3 predicciones de cada modelo
                        top3_main = np.argsort(res)[-3:][::-1]
                        top3_compare = np.argsort(res_compare)[-3:][::-1]
                        
                        print(f"\n🏆 Top 3 predicciones:")
                        print(f"   {model_type}:")
                        for i, idx in enumerate(top3_main, 1):
                            print(f"      {i}. {word_ids[idx]} ({res[idx]*100:.2f}%)")
                        print(f"   {compare_type}:")
                        for i, idx in enumerate(top3_compare, 1):
                            print(f"      {i}. {word_ids[idx]} ({res_compare[idx]*100:.2f}%)")
                    
                    print(f"{'='*50}\n")
                    
                    # Agregar palabra si supera el umbral
                    if res[predicted_idx] > threshold:
                        word_id = word_ids[predicted_idx].split('-')[0]
                        sent = words_text.get(word_id)
                        sentence.insert(0, sent)
                        text_to_speech(sent) # ONLY LOCAL (NO SERVER)
                
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []
            
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                
                # Mostrar tipo de modelo en la ventana
                model_label = f"[{model_type}] " + ' | '.join(sentence)
                cv2.putText(frame, solo_ascii(model_label), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                
                draw_keypoints(frame, results)
                cv2.imshow('Traductor LSA', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        video.release()
        cv2.destroyAllWindows()
        return sentence 

    
if __name__ == "__main__":
    # ========== CONFIGURACIÓN ==========
    import sys
    
    # Generar ruta del modelo TFLite basándose en MODEL_PATH
    TFLITE_PATH = MODEL_PATH.replace('.keras', '.tflite')
    
    print("=" * 50)
    print("EVALUADOR DE MODELOS DE SEÑAS")
    print("=" * 50)
    print(f"📂 Modelo Keras: {MODEL_PATH}")
    print(f"📂 Modelo TFLite: {TFLITE_PATH}")
    print("=" * 50)
    
    # Verificar qué modelos existen
    keras_exists = os.path.exists(MODEL_PATH)
    tflite_exists = os.path.exists(TFLITE_PATH)
    
    print(f"✓ Keras existe: {keras_exists}")
    print(f"✓ TFLite existe: {tflite_exists}")
    print("=" * 50)
    
    # Si se pasa argumento, usarlo para decidir qué modelo usar
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'tflite' and tflite_exists:
            print("🚀 Ejecutando con modelo TFLite...")
            evaluate_model(model_path=TFLITE_PATH, use_tflite=True)
        elif mode == 'keras' and keras_exists:
            print("🚀 Ejecutando con modelo Keras...")
            evaluate_model(model_path=MODEL_PATH, use_tflite=False)
        elif mode == 'compare':
            print("🔄 Modo comparación - Comparando ambos modelos en tiempo real...")
            if keras_exists and tflite_exists:
                # Ejecutar con comparación activada
                evaluate_model(model_path=MODEL_PATH, use_tflite=False, compare_mode=True)
            else:
                print("❌ Necesitas ambos modelos para comparar")
        else:
            print(f"❌ Modo '{mode}' no reconocido o modelo no existe")
            print("   Usa: python evaluate_model_tflite.py [keras|tflite|compare]")
    else:
        # Sin argumentos, preguntar qué hacer
        print("\nOpciones:")
        print("1. Probar modelo Keras")
        print("2. Probar modelo TFLite")
        print("3. Comparar ambos modelos")
        
        choice = input("\nElige una opción (1/2/3): ").strip()
        
        if choice == '1' and keras_exists:
            print("\n🚀 Ejecutando con modelo Keras...")
            evaluate_model(model_path=MODEL_PATH, use_tflite=False)
        elif choice == '2' and tflite_exists:
            print("\n🚀 Ejecutando con modelo TFLite...")
            evaluate_model(model_path=TFLITE_PATH, use_tflite=True)
        elif choice == '3':
            if keras_exists and tflite_exists:
                print("\n🔄 Comparando ambos modelos en tiempo real...")
                print("Las predicciones de ambos modelos se mostrarán simultáneamente")
                print("Presiona 'q' para salir\n")
                evaluate_model(model_path=MODEL_PATH, use_tflite=False, compare_mode=True)
            else:
                print("❌ Necesitas ambos modelos para comparar")
        else:
            print("❌ Opción no válida o modelo no existe")