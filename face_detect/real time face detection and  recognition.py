
# coding: utf-8

# # This script obtaining frames from camera,using mtcnn detecting faces,croping and embedding faces with pre-trained facenet and finally face recogition with pre-trained classifier.
# 
# 

# In[1]:



import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import random
import facenet
from scipy import misc
import sklearn

from sklearn.svm import SVC
import pickle

# In[2]:


#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#facenet embedding parameters

model_dir='model/20170512-110547/20170512-110547.pb'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=160 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."
margin=0

frame_interval=3 # frame intervals  


def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret


#obtaining frames from camera--->converting to gray--->converting to rgb
#--->detecting faces---->croping faces--->embedding--->classifying--->print

with tf.Graph().as_default():
	with tf.Session() as sess:
		facenet.load_model(model_dir)
		pnet, rnet, onet = detect_face.create_mtcnn(sess, 'model/')
		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

		with open('model/svm_classifier.pkl', 'rb') as infile:
			(model,labels) = pickle.load(infile)
					

		video_capture = cv2.VideoCapture(0)
		c=0
		
		while True:
			# Capture frame-by-frame
		
			ret, frame = video_capture.read()
			#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#print(frame.shape)
			
			timeF = frame_interval
			
			
			if(c%timeF == 0): #frame_interval==3, face detection every 3 frames
				
				find_results=[]
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
				
				if gray.ndim == 2:
					img = to_rgb(gray)
				
				bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
		
				
				
				nrof_faces = bounding_boxes.shape[0]#number of faces
				print('face num:',nrof_faces)
				print(bounding_boxes)
				if len(bounding_boxes) < 1:
					continue
				
				for face_position in bounding_boxes:
					face_position=face_position.astype(int)
					if face_position[1]<0 or face_position[3]<0 or face_position[0]<0 or face_position[2]<0:
						continue
					cropped = img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
					aligned = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
					prewhitened = facenet.prewhiten(aligned)
					
					face=np.array(prewhitened)
					
					face=face.reshape(-1,image_size,image_size,3)
					
					cv2.rectangle(frame, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0, 255, 0), 2)
					
					emb_data = sess.run([embeddings],
									feed_dict={images_placeholder:face, phase_train_placeholder:False})[0]
				
					predictions = model.predict_proba(emb_data)
					best_class_indices = np.argmax(predictions, axis=1)
					best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
					print(best_class_indices,best_class_probabilities)
					#print(predictions)
					print(labels[best_class_indices[0]],best_class_probabilities)
							
					#if predict=='pic_me':
					#	find_results.append('me')
					#elif predict=='pic_fan':
					#	find_results.append('fan')
					#else:
					#	find_results.append('others')
				
			
		
				cv2.putText(frame,'detected:{}'.format(find_results), (50,100), 
						cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
						thickness = 2, lineType = 2)
			c+=1

			# Display the resulting frame
		
			cv2.imshow('Video', frame)
		
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
		
		
		# When everything is done, release the capture
		
		video_capture.release()
		cv2.destroyAllWindows()
		
		