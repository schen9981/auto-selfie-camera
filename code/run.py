from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import cv2
import pickle
import cv2
from model import get_hog_features

# pretrained classifiers for face and eyes 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def detect_smile(pca, model, cropped_face):
  # get hog features
  cropped_face = cv2.resize(cropped_face, (100, 100))
  features = get_hog_features([cropped_face])

  # get reduced dim features
  features = pca.transform(features.reshape(1, -1))

  prediction = model.predict(features)

  return prediction


def save_image(path, img):
  return cv2.imwrite(path, img)

def crop_face(frame):
  ''' 
    takes in a frame,  detects the face, returns the 
    cropped matrix image of the face
  '''
  # preprocessing 
  faces = face_cascade.detectMultiScale(frame, 1.3, 5)

  if len(faces) != 0:
    for face in faces:
      # if face is found
      x = face[0]
      y = face[1]
      w = face[2]
      h = face[3]

      cropped = frame[x:x+w, y:y+h]
      return cropped
  else:
    return []

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

  # read in trained model
  filename = 'trained_model.sav'
  model = pickle.load(open(filename, 'rb'))

  # read in trained pca transformation
  pca_filename = 'trained_feature_transform.sav'
  pca = pickle.load(open(pca_filename, 'rb'))

  while True:
    # detect face in frame and crop
    # run face through model
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    cropped = crop_face(frame)

    if len(cropped) != 0:
      prediction = detect_smile(pca, model, cropped)

      print(prediction)

      # if model detects smile
      if prediction == 1:
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