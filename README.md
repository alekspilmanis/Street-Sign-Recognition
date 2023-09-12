# Street-sign-recognition

A program that classifies street signs from user uploaded images. 

classify.cpp can identify stop, ped xing, warning / advisory, road work ahead, and interchange / street signs. After an image is uploaded, classify.cpp will write onto the image the bounding rectangle of any street sign it detects, and will label it depending on which kind of sign it is. Can classify many street signs in one image.

find_masks.cpp was used only for development, to find the best HSV values that would work for each type of street sign. These values were then hardcoded into classify.cpp.
