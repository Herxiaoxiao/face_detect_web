
# coding: utf-8

# # This script processing images and training your own  face classifier.

# In[1]:



import tensorflow as tf
import numpy as np
import cv2

import os
import stat
from os.path import join as pjoin
import sys
import copy
import detect_face
import facenet
import matplotlib.pyplot as plt
from scipy import misc
import math
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier  

from sklearn.svm import SVC
import pickle


#parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

model_dir='model/20170512-110547/20170512-110547.pb'#model

image_size=160 #"Image size (height, width) in pixels."

use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= 20 # "Number of images to process in a batch."

#data_dir='train_dir/'#your own train folder
data_dir='test_dir/'
margin=0

def get_img_path_and_labels(data_dir):#get all image's paths and labels
	img_paths = []
	labels = []
	
	for guy in os.listdir(data_dir):
		j=0
		person_dir = pjoin(data_dir, guy)
		#if(len(os.listdir(person_dir))<35):
		#	continue
		for f in os.listdir(person_dir):
			img_paths.append(data_dir+guy+'/'+f)
			labels.append(guy)
			j+=1
		print(guy+':'+str(j))
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
        #img = cv2.imread(image,1)
        img = misc.imread(image, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        #print(image,bounding_boxes)
        if len(bounding_boxes) < 1:
          del(labels[image_paths.index(image)])
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        
        face_position=bounding_boxes[0].astype(int)
        
        if(face_position[1]<0 or face_position[3]<0 or face_position[0]<0 or face_position[2]<0):
            del(labels[image_paths.index(image)])
            image_paths.remove(image)
            print(face_position[1],face_position[3],face_position[0],face_position[2],image)
			
            continue
        cropped = img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
        aligned = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    print(labels)
    return img_list

def split_train_test(train,labels):
	return
	
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
		embedding_size = embeddings.get_shape()[1]
            
		# Run forward pass to calculate embeddings
		print('Calculating features for images')
		nrof_images = len(images)
		nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images /batch_size))
		emb_array = np.zeros((nrof_images, embedding_size))
		images_np=np.array(images).reshape(-1,image_size,image_size,3)
		for i in range(nrof_batches_per_epoch):
			start_index = i*batch_size
			end_index = min((i+1)*batch_size, nrof_images)

			feed_dict = { images_placeholder:images_np[start_index:end_index], phase_train_placeholder:False }
			emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
		
		#train/test split
		print('split train/test')
		train_x=emb_array
		train_y=np.array(labels)
		print(train_x.shape)
		print(train_y.shape)		
		#X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.6, random_state=42)
		#print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



		# train knn
		#print('train knn')
		#model = KNeighborsClassifier()  
		#model.fit(X_train, y_train)
		#joblib.dump(model, 'model/knn_classifier.model')
		#test knn
		#predict = model.predict(X_test)  
		#
		#accuracy = metrics.accuracy_score(y_test, predict)  
		#print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
		if(data_dir=='train_dir/'):
			# Train classifier
			print('Training classifier')
			model = SVC(kernel='linear', probability=True) 
			model.fit(train_x, train_y)
			
			# Saving classifier model
			temp=labels
			labels=list(set(temp))
			labels.sort(key=temp.index)
			print(labels)
			with open('model/svm_classifier.pkl', 'wb') as outfile:
				pickle.dump((model,labels), outfile)
			print('Saved classifier model to file model/svm_classifier.pkl')
		else:
			print('Testing classifier')
			with open('model/svm_classifier.pkl', 'rb') as infile:
				(model,labels) = pickle.load(infile)
			
			
			predictions = model.predict_proba(train_x)
			best_class_indices = np.argmax(predictions, axis=1)
			best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
			test=[]
			for i in range(len(train_y)):
				test.append(labels.index(train_y[i]))
				print(i,train_y[i],best_class_indices[i],best_class_probabilities[i])
			
			accuracy = np.mean(np.equal(best_class_indices, test))
			print('Accuracy: %.3f' % accuracy)
        #print(predictions)
        
        #l1 = ['b','c','d','b','c','a','a'] 
        #l2 = list(set(l1)) 
        #l2.sort(key=l1.index) 
        #print l2
		
		
		
		
		
		
		
		#python src/classifier.py TRAIN data/lfw/lfw_mtcnnpy_160 src/models/20170512-110547/20170512-110547.pb src/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset