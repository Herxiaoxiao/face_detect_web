from flask import Flask
from flask import render_template
from flask import request
import os, base64
from flask import jsonify
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
import face_detect.detect_face as detect_face
import random
import face_detect.facenet as facenet
from scipy import misc
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

model_dir='face_detect/model/20170512-110547/20170512-110547.pb'#"Directory containing the graph definition and checkpoint files.")
image_size=160 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

@app.route('/video')
def video():
    return render_template('video.html') 

@app.route('/image',methods=['POST'])
def get_image():
    
    imgData = base64.b64decode(request.form['img'])
    with open('temp/temp.jpg','wb') as f:
        f.write(imgData)
    bounding_boxes,best_class,best_class_probabilities=detect_face_()
    with open('temp/temp_face.jpg','rb') as f:
	    img_base64=str(base64.b64encode(f.read()),encoding = "utf-8")	
    if best_class==None:
        return jsonify({'success':0,})
    return jsonify({'success':1,'img': img_base64,'best_class':best_class,'best_class_probabilities':best_class_probabilities.tolist(),'bounding_boxes':bounding_boxes.tolist()})
	

def detect_face_():
    image='temp/temp.jpg'
    img = misc.imread(image, mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        return None,None,None
    i=0
    faces=np.zeros((len(bounding_boxes), image_size, image_size, 3))
    print(bounding_boxes)
    j=1
    for face_position in bounding_boxes:#左上角(0,0)
        face_position=face_position.astype(int)
        if face_position[1]<0 or face_position[3]<0 or face_position[0]<0 or face_position[2]<0:#face_position.any()<0
            continue
        cropped = img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
        aligned = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        prewhitened = facenet.prewhiten(aligned)
        
        faces[j-1,:,:,:]=np.array(prewhitened)
               
        cv2.rectangle(img, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0+i, 255, 0), 2)
        cv2.putText(img,str(j), (face_position[0]+5,face_position[1]+20), 
						cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255 ,255), 
						thickness = 1, lineType = 1)
        if i<=255:
            i+=150
        else:
            i=0
        j+=1
        cv2.imwrite('temp/temp_face.jpg',img)
    print(faces.shape)
    emb_data = sess.run([embeddings],
    				feed_dict={images_placeholder:faces, phase_train_placeholder:False})[0]
    
    predictions = model.predict_proba(emb_data)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    best_class=[]
    k=1
    for index in best_class_indices:
       best_class.append(str(k)+':'+labels[index])
       k+=1	   
    return bounding_boxes.astype(int),best_class,best_class_probabilities
	

	
if __name__ == '__main__':
    #from werkzeug.contrib.fixers import ProxyFix
    #app.wsgi_app = ProxyFix(app.wsgi_app) 
    #app.run(debug=True)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir)
            pnet, rnet, onet = detect_face.create_mtcnn(sess, 'face_detect/model/')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            with open('face_detect/model/svm_classifier.pkl', 'rb') as infile:
                (model,labels) = pickle.load(infile)
            app.run(host='172.24.41.55',port=5000,debug=True)