import cv2
import numpy as np

import tensorflow as tf
model = tf.keras.models.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while True:

    check,frame = video.read()
    frame = cv2.flip(frame,1)
    # Modify the input data by:

    # 1. Resizing the image

    img = cv2.resize(frame,(224,224))

    # 2. Converting the image into Numpy array and increase dimension

    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    # 3. Normalizing the image
    normalised_image = test_image/255.0

    # Predict Result
    prediction = model.predict(normalised_image)

    rock = int(prediction[0][0]*10)
    paper = int(prediction[0][1]*10)
    scissors = int(prediction[0][2]*10)
    txt = ''
    if rock >= 8 :
      txt = 'Rock'
    if paper >= 8:
      txt = 'Paper'
    if scissors >= 8:
      txt ='Scissors'

    print("Prediction :"," Rock:",rock," Paper:",paper," Scissors:",scissors)
    cv2.putText(frame,txt,(20,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow("Result",frame)
            
    key = cv2.waitKey(1)

    if key == 32:
        print("Closing")
        break

video.release()