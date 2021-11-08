from tensorflow.keras.models import load_model
from tensorflow.keras.preproccing import image
from tensorflow.keras.preproccing.image import ImageDataGenerator
from tensorflow.keras.preproccing.image import img_to_array
import numpy as np
import cv2
from playsound import playsound
import time

emotions_model = load_model("emotion.h5")
gesture_model - load_model("gesture.h5")
face_classififier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_label = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
gesture_label = ['loser', 'punch', 'super', 'victory']

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
labels = []
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faces_classifier.defectMultiScale(gray, 1.3, 5)

for (x, y , w, h) in faces:
    cv2.rectangle(frame,(x, y), (x + w, y + w), (255, 0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2, INTER_AREA)

    if np.sum([roi_gray]) !=0:
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi.axis = 0)
        preds = emotion_model.predict(roi)[0]
        label = emotion_label[preds.argmax()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    roi = frame[100:500, 100:500]
    img = cv2.resize(roi, (200, 200))
    img = image_img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = img.astype('float32')/255
    pred = np.argmax(gesture_model.predict(img))
