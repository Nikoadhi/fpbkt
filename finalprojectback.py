import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

print("Loading model")
global sess
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(
    sess
)
def get_model():
    global model
    model = tf.keras.models.load_model('finalprojectbangkit.h5')
    model._make_predict_function()
    print("Model Loaded")

global graph
graph = tf.compat.v1.get_default_graph

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    my_image = plt.imread(os.path.join('uploads', filename))
    my_image_re = resize(my_image, (1,150,150,3))
    

    with tf.Graph().as_default():
        
        model = tf.keras.models.load_model('finalprojectbangkit.h5')
        prediction = model.predict(my_image_re)
        
        
    
    return render_template('prediction.html', prediction=prediction)
    

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=80)