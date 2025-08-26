from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
from helpers import extract_keypoints, normalize_keypoints, there_hand, get_word_ids
from text_to_speech import text_to_speech

app = Flask(__name__)
CORS(app)

model = load_model("path_to_model.h5")
word_ids = get_word_ids("WORDS_JSON_PATH.json")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    frames = data.get("frames", [])
    if not frames:
        return jsonify({"error": "No frames provided"}), 400
    
    keypoints_seq = [extract_keypoints(frame) for frame in frames]
    normalized_keypoints = normalize_keypoints(keypoints_seq, target_length=15)
    
    prediction = model.predict(np.expand_dims(normalized_keypoints, axis=0))[0]
    predicted_word = word_ids[np.argmax(prediction)]
    
    confidence = prediction[np.argmax(prediction)]
    if confidence > 0.5:
        text_to_speech(predicted_word)
        return jsonify({"word": predicted_word, "confidence": float(confidence)})
    return jsonify({"word": None, "confidence": 0})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
