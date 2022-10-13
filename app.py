import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from flask import Flask, render_template, url_for, redirect,flash,request
from keras.models import load_model
import tensorflow.keras.utils as tku
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import joblib


model = load_model('od.h5')

app=Flask(__name__)
app.config['UPLOAD_FOLDER']=''


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST", "GET"])
def predict():

    if request.method == "POST":

      test_image = request.files['file']
      test_image.filename = "image.jpg"
      test_image.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(test_image.filename)))
      img = Image.open("image.jpg")
      img = np.asarray(img)
      size = (32, 32)
      img = cv2.resize(img, size)
      img=img.astype('float32')
      res = model.predict(img.reshape(1, 32, 32, 3)).argmax(axis=1)[0]
      names= ['Aeroplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
      prediction=names[res]
      
      
      pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')

      return render_template("index.html",prediction_text=f'{prediction}', input_image=pic1)
      


if __name__ == "__main__":
    app.run(debug=True)