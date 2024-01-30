from flask import Flask
from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
import os
from ai import recognition

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESIZED_FOLDER = os.path.join('static', 'resized')
MODEL_FOLDER = 'models'
MODEL_NAME = 'model03-extended.keras'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESIZED_FOLDER'] = RESIZED_FOLDER
app.config['MODEL_PATH'] = os.path.join(MODEL_FOLDER, MODEL_NAME)


@app.route("/")
def home():
    return render_template('base.html')


@app.route('/upload-sign', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['sign']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(path)
            label, prob, testpath = recognition.process_image(path)
            return render_template('result.html', result=label, img=path, prob=prob, testpath=testpath)
        else:
            return "File not found", 400
    else:
        return "GET not supported", 400
