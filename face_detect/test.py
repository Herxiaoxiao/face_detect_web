
# coding: utf-8

# # This script processing images and training your own  face classifier.

# In[1]:



import tensorflow as tf
import numpy as np
import cv2

import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import facenet
import matplotlib.pyplot as plt

from scipy import misc
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier  


#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#facenet embedding parameters

model_dir='model/20170512-110547/20170512-110547.pb'#"Directory containing the graph definition and checkpoint files.")

image_size=160 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."

data_dir='test/'#your own train folder
margin=32

def get_img_path_and_labels(data_dir):
	img_paths = []
	labels = []
	for guy in os.listdir(data_dir):
		person_dir = pjoin(data_dir, guy)
		for f in os.listdir(person_dir):
			img_paths.append(data_dir+guy+'/'+f)
			labels.append(guy)
        
	return img_paths,labels

def load_and_align_data(image_paths, image_size, margin):
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, 'model/')
  
    tmp_image_paths = image_paths.copy()
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        print(image,bounding_boxes)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        print(prewhitened)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images
image_paths ,labels=get_img_path_and_labels(data_dir)
images= load_and_align_data(image_paths, image_size, margin)

with tf.Graph().as_default():
	with tf.Session() as sess:
		facenet.load_model(model_dir)
		pnet, rnet, onet = detect_face.create_mtcnn(sess, 'model/')
		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		print('done')

		
		emb_data = sess.run([embeddings], 
									feed_dict={images_placeholder: images, phase_train_placeholder: False })[0]
				
		print(emb_data)
		
		print(image_paths ,labels)
		
		