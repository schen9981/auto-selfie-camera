from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import cv2
from facepipeline import 
import pickle

def detect_smile(mouth):
  # to be filled in with our model

def save_image(path, img):
  return cv2.imwrite(path, img)

def main():
  counter = 0
  total = 0
  threshold = 0.3
  print("Starting video stream...")
  vs = VideoStream(src=0).start()
  fileStream = False
  time.sleep(1.0)

  fps= FPS().start()
  cv2.namedWindow("test")

  filename = 'trained_model.sav'
  model = pickle.load(open(filename, 'rb'))

  while True:
    # detect face in frame and crop
    # run face through model
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    prediction = detect_smile(frame)

    # if model detects smile
    if prediction > threshold:
      # accumulate num of frames
      counter += 1

    # if smile held for 15 frames, take selfie
    if counter >= 15:
      frame = vs.read()
      time.sleep(.3)
      frame2= frame.copy()
      
      img_name = "detected_smile_{}.png".format(total)
      total += 1

      save_image('../results/{}'.format(img_name), frame)

      print("{} captured with likelihood {}".format(img_name, prediction))

      cv2.imshow("Frame", frame)
      fps.update()

      counter = 0

    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
  main()