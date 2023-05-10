import cv2
import numpy as np
import face_recognition

imgisha = face_recognition.load_image_file('ImagesBasic/isha.jpg')
imgisha = cv2.cvtColor(imgisha,cv2.COLOR_BGR2RGB)
imgishatest = face_recognition.load_image_file('ImagesBasic/ishatest.jpg')
imgishatest = cv2.cvtColor(imgishatest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgisha)[0]
encodeisha = face_recognition.face_encodings(imgisha)[0]
cv2.rectangle(imgisha,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoctest = face_recognition.face_locations(imgishatest)[0]
encodeishatest = face_recognition.face_encodings(imgishatest)[0]
cv2.rectangle(imgishatest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeisha],encodeishatest)
faceDis = face_recognition.face_distance([encodeisha],encodeishatest)
print(results,faceDis)
cv2.putText(imgishatest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('isha',imgisha)
cv2.imshow('ishatest',imgishatest)
cv2.waitKey(0)