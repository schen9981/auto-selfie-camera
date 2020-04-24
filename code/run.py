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
  fileStream = True
  # time.sleep(1.0)

  fps= FPS().start()
  # cv2.namedWindow("test")

  # read in trained model
  filename = 'trained_model.sav'
  model = pickle.load(open(filename, 'rb'))

  # read in trained pca transformation
  pca_filename = 'trained_feature_transform.sav'
  pca = pickle.load(open(pca_filename, 'rb'))

  while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cropped = crop_face(grey_frame)

    if len(cropped) != 0:
      prediction = int(detect_smile(pca, model, cropped)[0])

      print(prediction)
      if prediction == 1:
        counter += 1
      
      if counter >= 10:
        frame = vs.read()
        # time.sleep(.3)
        frame2= frame.copy()
        img_name = "detected_smile_{}.png".format(total)
        save_image('../results/{}'.format(img_name), frame)
        print("{} written!".format(img_name))
        counter = 0
        total += 1

    cv2.imshow("Frame", frame)
    fps.update()

    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break

  fps.stop()
  cv2.destroyAllWindows()
  vs.stop()

if __name__ == '__main__':
  main()