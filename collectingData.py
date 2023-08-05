import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 400
folder = 'Data/C'
counter = 0
while True:
       succes, img = cap.read()
       hands, img = detector.findHands(img)
       if hands:
              hand = hands[0]
              x, y, w, h = hand['bbox']
              imgWhite = np.ones((imgsize, imgsize, 3), np.uint8)*255
              imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

              aspectRatio = h/w
              if aspectRatio > 1:
                  k = abs(int(imgsize/h))
                  newW = math.ceil(k*w)
                  if newW > imgsize:
                      newW = imgsize
                  imgResize = cv2.resize(imgCrop, (newW, imgsize))
                  wGap = math.ceil((imgsize-newW)/2)
                  imgWhite[:, wGap:newW+wGap] = imgResize
              else :
                  k = abs(int(imgsize / w))
                  newH = math.ceil(k * h)
                  if newH > imgsize:
                      newH = imgsize
                  imgResize = cv2.resize(imgCrop, (imgsize, newH))
                  HGap = math.ceil((imgsize - newH) / 2)
                  imgWhite[HGap:newH + HGap, :] = imgResize
              cv2.imshow("imgCrop", imgCrop)
              cv2.imshow("imgWhite", imgWhite)

       cv2.imshow("image",img)
       key = cv2.waitKey(1)
       if key == ord("s"):
           counter += 1
           cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
           print(counter)

