### This program is used to check all of the images within the red, yellow, and gree
### traffic light directories such that the user can remove any images from directories
### that they did not label correctly. Helps to prevent any bad data in the training set

import cv2 as cv
import os
import TrafficLightExtractor

# Uses input from the user to remove any traffic lights from the training set that 
# were improperly labeled
def removeBadData(nameOfDirectory, nameOfLabel):
    ## Prompt the user to keep the image in the directory if it was correctly labeled
    ## or remove it if it is incorrect
    print("The following images have been labeled as " + nameOfLabel + ".")
    print("For each image, determine whether the label is correct")
    print("Type c if the label is correct")
    print("Type i if the label is incorrect")
    print("Type q to quit")
    
    # Get path to image file directory
    currentPath = os.getcwd()
    directoryPath = currentPath + nameOfDirectory

    # Get all the images in the directory
    imageNames= TrafficLightExtractor.getFilesInDirectory(nameOfDirectory)
    # For each image stored in the directory
    for imageName in imageNames:
        # Read the image
        trafficLightImage = cv.imread(imageName)
        # Show the image to the user and get their input
        cv.imshow(nameOfLabel, trafficLightImage)
        userInput = cv.waitKey(0)
        # If label is incorrect, delete the image
        if userInput == ord('i'):
            print(imageName)
            os.remove(imageName)
        # If label is correct, skip to next image
        elif userInput == ord('c'):
            continue
        # If user wants to quit, exit the program
        elif userInput == ord('q'):
            exit(1)
        

if __name__ == "__main__":
    ## Get rid of all incorrectly-labeled images in the red, yellow, and green
    ## light directories and the didrectory of nontraffic lights
    nonTrafficLightDirectory = "/not_traffic_lights"
    removeBadData(nonTrafficLightDirectory, "not traffic lights")
    # Red light directory
    redLightDirectory = "/red_traffic_lights"
    removeBadData(redLightDirectory, "red traffic lights")
    # Yellow light directory
    yellowLightDirectory = "/yellow_traffic_lights"
    removeBadData(yellowLightDirectory, "yellow traffic lights")
    # Green light directory
    greenLightDirectory = "/green_traffic_lights"
    removeBadData(greenLightDirectory, "green traffic lights")