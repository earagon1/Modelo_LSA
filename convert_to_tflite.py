#import tensorflow as tf
#from tensorflow import keras
#from constants import MODEL_PATH
#import os

#import numpy as np

# Cargar el modelo .keras
#model = keras.models.load_model(MODEL_PATH)
#model = tf.keras.models.load_model(MODEL_PATH)

# Convertir a TFLite con opciones avanzadas
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.target_spec.supported_ops = [
#    tf.lite.OpsSet.TFLITE_BUILTINS,
#    tf.lite.OpsSet.SELECT_TF_OPS
#]
#converter._experimental_lower_tensor_list_ops = False  # Desactiva el manejo inestable


# Convertir con optimización para móviles
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float32]  # Mantener float32 para precisión


#tflite_model = converter.convert()

# Guardar
#tflite_path = os.path.join(os.path.dirname(MODEL_PATH), "model.tflite")
#with open(tflite_path, 'wb') as f:
#    f.write(tflite_model)

# Guardar
#with open('model.tflite', 'wb') as f:
#    f.write(tflite_model)

# Verificar el modelo
#interpreter = tf.lite.Interpreter(model_content=tflite_model)
#interpreter.allocate_tensors()

#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()

#print(f"Input shape: {input_details[0]['shape']}")  # Debe ser [1, 15, 126]
#print(f"Output shape: {output_details[0]['shape']}")  # Debe ser [1, 10]

#print(f"Modelo convertido y guardado en: {tflite_path}")

# Test con entrada dummy
#test_input = np.random.randn(1, 15, 126).astype(np.float32)
#interpreter.set_tensor(input_details[0]['index'], test_input)
#interpreter.invoke()
#output = interpreter.get_tensor(output_details[0]['index'])
#print(f"Test output shape: {output.shape}")
#print(f"Test output sum: {output.sum():.3f}")  # Debe ser ~1.0


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from constants import MODEL_PATH

TFLITE_PATH = 'model.tflite'

def convert_model(debug_tensorlist_workaround: bool = False):
    print("🔍 Cargando modelo Keras...")
    model = keras.models.load_model(MODEL_PATH)
    model.summary()

    print("\n⚙️ Configurando conversor TFLite (float32, SELECT_TF_OPS)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    # Mantener float32 para depurar diferencias numéricas
    # (no seteamos representative_dataset ni types)

    # Workaround si aparece FlexTensorList*
    if debug_tensorlist_workaround:
        converter._experimental_lower_tensor_list_ops = False  # <- solo si es necesario

    # Convertir
    print("\n🔄 Convirtiendo...")
    tflite_model = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Guardado: {os.path.abspath(TFLITE_PATH)}")

    # Verificación con Interpreter
    print("\n🔍 Verificando modelo convertido...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"📥 Input: name={input_details[0]['name']}, shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"📤 Output: name={output_details[0]['name']}, shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Smoke test
    x = np.random.randn(1, 15, 126).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]['index'])
    print(f"🧪 Predicción dummy: shape={y.shape}, sum={float(y.sum()):.4f}")

if __name__ == "__main__":
    convert_model(debug_tensorlist_workaround=False)
