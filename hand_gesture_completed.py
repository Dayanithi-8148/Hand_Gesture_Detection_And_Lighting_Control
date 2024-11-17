import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import serial
import time

comport = 'COM17'
baud_rate = 9600  
ser = serial.Serial(comport, baud_rate, timeout=1)
time.sleep(2)

total=0
def led(total):
    command = str(total) + '\n'
    ser.write(command.encode())
model = load_model('./mp_hand_gesture.h5')

with open('./gesture.names', 'r') as f:
    classNames = f.read().split('\n')

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)

    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            prediction = np.asarray(prediction)  
            classID = np.argmax(prediction)
            
            if classID < len(classNames):
                className = classNames[classID]
            else:
                className = "Unknown Class"
            
            if className == 'Green Light On':
                total = 1  
            elif className == 'Green Light Off':
                total = 2
            elif className == 'Yellow Light Off':
                total = 3
            elif className == 'Yellow Light On':
                total = 4
            elif className == 'Red Light On':
                total = 5
            elif className == 'Red Light Off':
                total = 6
            print(total)
            led(total)

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
