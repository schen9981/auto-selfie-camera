## Code

This directory contains all the python files used to process raw data, train the smile detector, and integrate a livestream
face video. A breakdown of the functionality of each file is below.

### Haar Cascade XML Files 

The two XML files, haarcascade_frontalface_alt.xml and haarcascade_frontalface_default.xml are the pretrained inputs 
to a Haar Cascade algorithm for frontal face detection. These were taken from OpenCV's free library, and are used 
to run the model that will detect the location of the face in an image.

### facepipeline.py

This python file is run to generate the images in the data/genki4k/files/cropped/ directory. This file will read
in each image from data/genki4k/files/original/, run it through a Haar Cascade face detection (with the xml files),
and using the outputted rectangle coordinates, appropriately crop the image to the rectangle to only contain the face pixels.
The smile detection model is then trained on the cropped images generated. 

### preprocess.py

This python file contains the read_raw_data function, which reads in the cropped images and 
appropriately format them to be stored, along with the labels in the labels.txt file. There is also a train_test_split
function that will split the data and the labels into a train and test set, with a 80-20 ratio. 

### model.py

This python file contains all function that will generate and train the model to be used to detect smiles in images. More 
detail on the specifics of the model can be found the the README of the whole project repo. 

### run.py

This file contains the main driver methods for this application, which will read in the frames of the camera and 
run it through our smile detection model. As soon as a smile is detected (for a certain number of frames), an image
is saved as a 'selfie'.
