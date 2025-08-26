# Verifica tu modelo en Python
import tensorflow as tf
import numpy as np
from constants import *

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")  # Debe ser [1, 15, 126]
print(f"Output shape: {output_details[0]['shape']}") # Debe ser [1, num_clases]

# Prueba con datos dummy
test_input = np.random.randn(1, 15, 126).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"Output: {output}")