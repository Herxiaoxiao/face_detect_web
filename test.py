


import cv2

import numpy as np

import os

from scipy import misc



image='temp/temp.jpg'
img = misc.imread(image, mode='RGB')
bounding_boxes=[[128.89334685,290.71496652,191.75738479,370.95397262],[280.03894717,188.70413291,474.71068705,432.28776578]]
bounding_boxes=np.asarray(bounding_boxes)
print(bounding_boxes)

i=0
j=1
for face_position in bounding_boxes:
    face_position=face_position.astype(int)
    if face_position[1]<0 or face_position[3]<0 or face_position[0]<0 or face_position[2]<0:
        continue
               
    cv2.rectangle(img, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0+i, 255, 0), 2)
    cv2.putText(img,str(j), (face_position[0]+5,face_position[1]+20), 
					cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0+i, 255 ,0), 
					thickness = 1, lineType = 1)
    if i<=255:
        i+=150
    else:
        i=0
    j+=1
cv2.imshow('temp/temp_face1.jpg',img)
cv2.waitKey(0)