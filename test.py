import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/Keras_model.h5", "model/labels.txt")
offset = 20
imgSize = 400
folder = 'Data/C'
counter = 0
labels = ["A", "B", "C"]
while True:
    succes, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = abs(int(imgSize / h))
            newW = math.ceil(k * w)
            if newW > imgSize:
                newW = imgSize
            imgResize = cv2.resize(imgCrop, (newW, imgSize))
            wGap = math.ceil((imgSize - newW) / 2)
            imgWhite[:, wGap:newW + wGap] = imgResize
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)

        else:
            k = abs(int(imgSize / w))
            newH = math.ceil(k * h)
            if newH > imgSize:
                newH = imgSize
            imgResize = cv2.resize(imgCrop, (imgSize, newH))
            HGap = math.ceil((imgSize - newH) / 2)
            imgWhite[HGap:newH + HGap, :] = imgResize
        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("image", img)
    cv2.waitKey(1)
