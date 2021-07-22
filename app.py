import os
import json
import pickle
import numpy as np
import eda_utils as et
from flask import Flask, redirect, url_for, request, render_template

UPLOAD_FOLDER = '.'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = False

# ---------------------------- load model
filename = "models/ada.sav"
ada = pickle.load(open(filename, 'rb'))

@app.route('/')
def svc_default():
    return "It is connected~"

@app.route('/cat')
def meow_default():
    return "Meow"

@app.route('/predict/json', methods = ['GET','POST'])
def predict_json_dtree():
    print("detected request attempt.")
    if request.method == 'POST':
        obj = request.get_json()
        input_frames = np.array(obj["input_frames"])
        processed_df = et.map_inputframe_modelframe(input_frames)
        result = list(ada.predict(processed_df))
        resp = {}
        resp['prediction'] = [int(i) for i in result]
        return resp

if __name__ == '__main__':
	#app.run(host="127.0.0.1")
	app.run(host="0.0.0.0")
