import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

def read_image(imgname):

    image = np.asarray(Image.open(imgname).resize((100,75)))
    return image

def predict_x(x):
    
    model = keras.models.load_model('traffic_classifier.h5')
    p = model.predict(x)
    return p

def classify(imgname):
    

    image = read_image("static/uploads/"+imgname)
    # x = image.reshape(128, 128, 3)
    # x= tf.keras.utils.img_to_array(x)
    # x = tf.expand_dims(x, 0)
    # x = (x - 159.88411714650246)/46.45448942251337
    img = tf.keras.utils.load_img(
    "static/uploads/"+imgname, target_size=(30, 30))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    p = predict_x(img_array)
    pr = p[0].tolist()
    return pr