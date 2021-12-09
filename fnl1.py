from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2 
from playsound import playsound
import time

emotion_model = load_model("emotions.h5")
gesture_model = load_model("gesture.h5")
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_label = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
gesture_label = ['loser', 'punch', 'super', 'victory']
    
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y , w, h) in faces:
        cv2.rectangle(frame,(x, y), (x + w, y + w), (255, 0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)

        if np.sum([roi_gray]) !=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis = 0)

            preds = emotion_model.predict(roi)[0]
            label = emotion_label[preds.argmax()]

            start = time.clock()
            print(start)

            if int(start) % 10 == 0:
                if label == "Angry":
                    playsound("reactions/anger.mp3")
                elif label == "Fear":
                    playsound("reactions/fear.mp3")
                elif label == "Happy":
                    playsound("reactions/happy.mp3")
                elif label == "Neutral":
                    playsound("reactions/neutral.mp3")
                elif label == "Sad":
                    playsound("reactions/sad.mp3")
                elif label == "Surprise":
                    playsound("reactions/surprise.mp3")
            label_position = (x,y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame, 'No Face Found', (20 , 60), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),3)
    
    ret1, frame1 = cap.read()
    cv2.rectangle(frame1, (100, 100), (500, 500), (255, 255, 255), 2)
    roi1 = frame1[100:500, 100:500]
    img1 = cv2.resize(roi1, (200, 200))
    img1 = image.img_to_array(img1)
    img1 = np.expand_dims(img1, axis = 0)
    img1 = img1.astype('float32')/255
    pred = np.argmax(gesture_model.predict(img1))
    color =(0,0, 255)

    start = time.clock()
    print(start)

    if int(start) % 10 == 0:
        if pred == 0:
            playsound("reactions/loser.mp3")
        elif pred == 1:
            playsound("reactions/punch.mp3")
        elif pred == 2:
            playsound("reactions/super.mp3")
        elif pred == 3:
            playsound("reactions/victory.mp3")

    cv2.putText(frame, gesture_label[pred], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
