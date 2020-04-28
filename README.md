## Computer Vision Final Project: Automated Selfie Camera

Gabby Asuncion, Sophia Chen, Kelvin Yang

### Overview 
This application is a live camera app that will capture an image
when it detects the subject smiling. There are 2 core components:

1. the smile detection model
2. the live camera feed

### Smile Detection Model

At the core of the smile detection model, we have a support vector classifier with a linear kernel, with a 
margin of error of 5. Once the image data is read in (after being cropped to the face, resized to 100 by 100, 
and converted to greyscale), the HOG (Histogram of Oriented Gradients) features are extracted from the images, with 
15 orientations, 10 cells per block, and 10 pixels per cell. However, because this extract a substantial number
of features and thus creating a high dimensional feature space, we apply Principle Component Analysis (PCA) to
transform the data into a smaller, more manageable feature space, and pinpoint the features that contribute
most to effective classification. Once the input images have been transformed as described, they are fed into
an SVM and trained. 

Because we only had 4000 images, some of which were not perfectly oriented, we want utilized k-fold cross validation
in order to generate more train-test sets to determine the performance of the model. In this project, we carried 
out 5-fold cross validation, and observed the scores in each fold to determine the optimal parameters for 
our final trained model.

Once the parameters were appropriately tuned and the accuracy was evaluted, the final trained SVM model, along with
the tuned PCA transformation, were pickled and saved to be read into the driver program for the live camera feed. 


### Live Camera Feed 

The live camera feed is implemented using a VideoStream package, which, once initialized, will read in
frames that are represented as pixel matrices. Frames are continuously read in and fed into the pre-trained
model face detection model, in order to crop the frame/image appropriately to be fed into the smile detection
model. If a smile is detected for at least 10 frames, the frame is saved as a .png image in the /results directory. 


