# This program is used to segment the traffic lights into four distinct groups, 
# one group for each color in (red, yellow, green) and a fourth group for
# traffic lights with unobservable light colors or objects that were 
# misidentified as traffic lights
import cv2 as cv
import TrafficLightExtractor
import os

# Shows each traffic light to the user and has them label its color
# The traffic light image is then placed into a directory reserved
# for that given color
def sortbyLightColor(directoryOfTrafficLights):
    listOfDirectories = []
    # Store current path for later use
    currentPath = os.getcwd()
    ## Create separate directories to store images of the three 
    ## different colors of traffic lights
    redLightFolder = "/red_traffic_lights2"
    redLightPath = currentPath + redLightFolder
    listOfDirectories.append(redLightFolder)

    yellowLightFolder = "/yellow_traffic_lights2"
    yellowLightPath = currentPath + yellowLightFolder
    listOfDirectories.append(yellowLightFolder)

    greenLightFolder = "/green_traffic_lights2"
    greenLightPath = currentPath+ greenLightFolder
    listOfDirectories.append(greenLightFolder)
    # Folder for traffic lights where color isnt visible or simply arent
    # traffic lights
    nonTrafficLightFolder = "/not_traffic_lights2"
    nonTrafficLightPath = currentPath + nonTrafficLightFolder
    listOfDirectories.append(nonTrafficLightFolder)

    # Create new directory to store images of objects within the four
    # groups
    for directoryName in listOfDirectories:
        TrafficLightExtractor.createDirectory(directoryName)


    # Get a list of all the images of traffic lights
    trafficLightImages = TrafficLightExtractor.getFilesInDirectory(directoryOfTrafficLights)

    ## Prompt the user to label each image as either a red, green, or yellow traffic
    ## light or an object that is not a traffic light
    print("Label the following traffic lights according to the color of their lights")
    print("Type r for red traffic light")
    print("Type y for yellow traffic light")
    print("Type g for green traffic light")
    print("Type n if the image is not of a traffic light or the color of the light can \
          not be seen")
    print("Type q at any time to quit")
    # Create counter variables for each group
    greenCounter = 0
    redCounter = 0
    yellowCounter = 0
    otherCounter = 0
    for imageName in trafficLightImages:
        # Read image from image file path
        trafficLightImage = cv.imread(imageName)
        # Show the image to the user
        cv.imshow("Label Traffic Light Color", trafficLightImage)
        # Read the input label they give to the image
        userInput = cv.waitKey(0)
        ## Save the image in the directory specified by user
        # If user wants to quit, exit the program
        if userInput == ord('q'):
            exit(1)
        # Put the image in their respective folders
        if userInput == ord('g'):
            fileName = "greenLight" + str(greenCounter) + ".jpg"
            os.chdir(greenLightPath)
            cv.imwrite(fileName, trafficLightImage)
            greenCounter += 1
        elif userInput == ord('y'):
            fileName = "yellowLight" + str(yellowCounter) + ".jpg"
            os.chdir(yellowLightPath)
            cv.imwrite(fileName, trafficLightImage)
            yellowCounter += 1
        elif userInput == ord('r'):
            fileName = "redLight" + str(redCounter) + ".jpg"
            os.chdir(redLightPath)
            cv.imwrite(fileName, trafficLightImage)
            redCounter +=1
        elif userInput == ord('n'):
            fileName = "NotTrafficLight" + str(otherCounter) + ".jpg"
            os.chdir(nonTrafficLightPath)
            cv.imwrite(fileName, trafficLightImage)
            otherCounter += 1
        # If the user entered a q, exit the program
        elif userInput == ord('q'):
            exit(1)
            
if __name__ == "__main__":
    sortbyLightColor("/Traffic_Lights")