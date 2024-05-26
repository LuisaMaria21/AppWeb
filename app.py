from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from flask_socketio import SocketIO, emit
import base64
from PIL import Image # deben instalar pillow

app = Flask(__name__)
socketio = SocketIO(app)

# Cargar el modelo
model = tf.keras.models.load_model('model/donalex74%.h5')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('message')
def handle_message(data):
    # recibimos la imagen en base64
    base64_string = ""

    if "data:image" in data:
        base64_string = data.split(",")[1] # separamos el base64 porque llega en un formato data:image,asdfasdfasdf

    image_bytes = base64.b64decode(base64_string) # decodificamos el base64
    img = Image.open(io.BytesIO(image_bytes)) # lo volvemos una imagen real, tienen que mirar si el modelo puede recibir la imagen asi o si funciona el codigo que sigue
    # Desde aqui no se si funcione porque no tengo como probar esta parte
    npimg = np.frombuffer(img, np.uint8)
    cv2img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
    # Procesar la imagen (esto depende de tu modelo)
    cv2img = cv2.resize(cv2img, (224, 224))  # Ajusta el tamaño según tu modelo
    cv2img = cv2img / 255.0  # Normalizar
    cv2img = np.expand_dims(cv2img, axis=0)

    # Realizar la predicción
    y_predicted = model.predict(cv2img)
    predicted_label = np.argmax(y_predicted[0])  # Obtener la clase con mayor probabilidad
    classes = ['cuchillo', 'tenedor', 'cuchara']  # Ajusta esto según las clases de tu modelo
    prediction_text = classes[predicted_label]

    socketio.emit('message', {'prediction': prediction_text, 'class_prediction': classes}) # emitimos el mensaje

    print('received message: ' + 'ok')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Leer la imagen del request
#         file = request.files['image'].read()
#         npimg = np.frombuffer(file, np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
#         # Procesar la imagen (esto depende de tu modelo)
#         img = cv2.resize(img, (224, 224))  # Ajusta el tamaño según tu modelo
#         img = img / 255.0  # Normalizar
#         img = np.expand_dims(img, axis=0)

#         # Realizar la predicción
#         y_predicted = model.predict(img)
#         predicted_label = np.argmax(y_predicted[0])  # Obtener la clase con mayor probabilidad
#         classes = ['cuchillo', 'tenedor', 'cuchara']  # Ajusta esto según las clases de tu modelo
#         prediction_text = classes[predicted_label]
        
#         return jsonify({'prediction': prediction_text, 'class_prediction': classes})
#     except Exception as e:
#         return jsonify({'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
