## Data

Raw data was from the GENKI 4K dataset, collected by the Machine Perception Laboratory, University of California,
San Diego. This dataset contains 4000 JPG images of faces, which are located in data/genki4k/files/original. Each of
these images has a labeled expression in the corresponding row of labels.txt, located in data/genki4k/. A breakdown
of the contents of each of the data directories and files are below:

### Raw Face Images 

Location: data/genki4k/files/original/

Description: 4000 JPG images of faces with expressions. 

### Expression Labels

Location: data/genki4k/

Description: A text file with 4000 lines, where the nth line of the file corresponds to the label for the 
nth image in the raw face images directory. A 1 indicates a smile, while a 0 indicates non-smile. Additional 
data in the lines indicates face pose (yaw, pitch, roll) in radians. 

### Cropped Face Images

Location: data/genki4k/files/cropped/

Description: 4000 PNG images of cropped faces, based on the raw face images in the 'original' directory. 
The original images were fed into a Haar Cascade face detection model, and the images were then cropped based on
the rectangle outputted that contained the detected face. 
