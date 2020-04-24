import numpy as np
import cv2 
import os
from PIL import Image

# pretrained classifiers for face and eyes 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detection(og_path, cropped_dir):
    ''' 
        retrieves img and finds the faces in the image as a rectangle (assumes only one face)
    '''
    # get list of image files
    og_imgs = sorted(os.listdir(og_path))

    for i, img in enumerate(og_imgs):
        curr_path = os.path.join(og_path, img)
        og = Image.open(curr_path).convert('LA')

        numpy_image = cv2.imread(curr_path)
        
        # find face in images
        # if face found, returns the position of detected faces as a rectangle (x, y, w, h)
        faces = face_cascade.detectMultiScale(numpy_image, 1.3, 5)

        filename = 'cropped_' + img[:-4] + '.png'
        write_path = os.path.join(cropped_dir, filename)
        if len(faces) != 0:
            for face in faces:
                cropped_img = img_crop(og, faces[0])
                cropped_img.save(write_path)
                # c(write_path, cropped_img)
        else:
            # write original image
            og.save(write_path)


def img_crop(og_img, face):
    xdelta = face[2]
    ydelta = face[3]

    crop_box = [face[0] - xdelta, face[1] - ydelta, face[0] + face[2] + xdelta, face[1] + face[3] + ydelta]

    return og_img.crop(crop_box)

def main():
    og_dir = '../data/genki4k/files/original'
    cropped_dir = '../data/genki4k/files/cropped'
    face_detection(og_dir, cropped_dir)
    

if __name__ == '__main__':
    main()




    



