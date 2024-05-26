from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Cargar el modelo
model = tf.keras.models.load_model('model/donalex74%.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Leer la imagen del request
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Procesar la imagen (esto depende de tu modelo)
        img = cv2.resize(img, (224, 224))  # Ajusta el tamaño según tu modelo
        img = img / 255.0  # Normalizar
        img = np.expand_dims(img, axis=0)

        # Realizar la predicción
        y_predicted = model.predict(img)
        predicted_label = np.argmax(y_predicted[0])  # Obtener la clase con mayor probabilidad
        classes = ['cuchillo', 'tenedor', 'cuchara']  # Ajusta esto según las clases de tu modelo
        prediction_text = classes[predicted_label]
        
        return jsonify({'prediction': prediction_text, 'class_prediction': classes})
    except Exception as e:
        return jsonify({'error': str(e)})

@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)

if __name__ == '__main__':
    socketio.run(app, debug=True)
