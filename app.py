from flask import Flask, flash, request, jsonify, render_template, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images/'
RESULT_FOLDER = 'static/images/res'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO

import socket 
import io 


#function
def process_image(img):
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

def get_classes(file):
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

def draw(image, boxes, scores, classes, all_classes):
    count = 0

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        
        if all_classes[cl] != 'person':
            continue
            
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        count += 1
        # print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        # print('box coordinate x,y,w,h: {0}'.format(box))
    return count

def detect_image(image, yolo, all_classes):
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    # print('time: {0:.2f}s'.format(end - start))

    count = 0
    if boxes is not None:
        count = draw(image, boxes, scores, classes, all_classes)
    return image, count

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#variable
yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'
all_classes = get_classes(file)

#route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            image = cv2.imread(path)
            image, count = detect_image(image, yolo, all_classes)      
            cv2.imwrite(app.config['RESULT_FOLDER'] + file.filename, image)
            fileDetected = os.path.join(app.config['RESULT_FOLDER'] , file.filename)
            return redirect(url_for('show_result', name=file.filename, count=count))
    return render_template("upload.html")

@app.route('/result/<name>,<count>')
def show_result(name, count):
    return render_template("result.html", person_image = name, countPerson = count)    

@app.route('/live') 
def live():
   return render_template('live.html') 

def gen():
   vc = cv2.VideoCapture(0)
   while True: 
       rval, frame = vc.read() 
       image, count = detect_image(frame, yolo, all_classes)
       cv2.imwrite(app.config['RESULT_FOLDER'] + '/picFromCam.jpg', image)
       yield (b'--frame\r\n' 
              b'Content-Type: image/jpeg\r\n\r\n' + open(app.config['RESULT_FOLDER'] + '/picFromCam.jpg', 'rb').read() + b'\r\n') 

@app.route('/video_feed') 
def video_feed():
   return Response(gen(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame') 



#start at port 5000
if __name__ == '__main__':
    app.run(threaded=True, port=5000)