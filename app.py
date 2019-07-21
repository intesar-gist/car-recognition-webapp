#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from flask import Flask, render_template, jsonify, request

import trained_model as tm
import ModelType
app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = ['jpg', 'jpeg']

carMakeModel, carMakeGraph, carMakeLabel = tm.load_model(ModelType.ModelType.car_make)
carModelModel, carModelGraph, carModelLabel = tm.load_model(ModelType.ModelType.car_model)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/car_make")
def car_make():
    return render_template("car_make.html")


@app.route("/car_model")
def car_model():
    return render_template("car_model.html")

@app.route('/upload/<string:model_type>', methods=['POST'])
def upload(model_type):
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            now = datetime.now()
            filename = os.path.join(app.config['UPLOAD_FOLDER'], "%s.%s" % (now.strftime("%Y-%m-%d-%H-%M-%S-%f"), file.filename.rsplit('.', 1)[1]))
            file.save(filename)

            if ModelType.ModelType.car_model.name == model_type:
                prediction = tm.predict(carModelGraph, carModelModel, filename, carModelLabel)
            elif ModelType.ModelType.car_make.name == model_type:
                prediction = tm.predict(carMakeGraph, carMakeModel, filename, carMakeLabel)
            else:
                os.remove(filename)
                return jsonify("wrong model type")

            os.remove(filename)

            return jsonify(prediction)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(debug=True)
