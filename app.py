from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
socketio = SocketIO(app)
modelo = load_model('C:\\Users\\User\\OneDrive - Universidad Autonoma de Occidente\\Documentos\\UAO\\Semestre 10\\PDI\\flask_app\\Model\\donalex72.h5')

# Definir las clases
CLASES = ['0', '1', '2']

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Ajusta el tamaño según tu modelo
    img = img / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)
    return img

def predict_class(frame):
    preprocessed_frame = preprocess_image(frame)
    prediction = modelo.predict(preprocessed_frame)
    predicted_label = np.argmax(prediction)
    predicted_class = CLASES[predicted_label]  # Obtener la clase correspondiente
    return predicted_class

def camera_stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_prediction')
def get_prediction():
    # Esto es un ejemplo, necesitas ajustarlo para tu flujo de trabajo
    return jsonify({
        'prediccion': '0',  # Ejemplo de predicción
        'classes': CLASES
    })

@socketio.on('message')
def handle_message(data):
    frame_data = data.split(',')[1]
    frame_bytes = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

    predicted_class = predict_class(frame)
    emit('message', {'prediccion': predicted_class, 'classes': CLASES})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
