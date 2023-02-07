### This program was used to extract all the images that contain traffic lights within
###  a given input video
### First, the user is asked to go through a video frame by frame and identify which 
### frames have traffic lights
import cv2 as cv
import os
import TrafficLightExtractor


if __name__ == "__main__":
    # While frames are still being read from the traffic capture video
    inputFileName ='DetectionVideos/DrivingThroughSeattle.mp4'
    # Load the traffic video
    trafficCapture = cv.VideoCapture(inputFileName)

    # Create folder to store traffic-light containing images in
    ImagesWithTLDirectory = "/ImagesWithTrafficLights2"
    TrafficLightExtractor.createDirectory(ImagesWithTLDirectory, deleteExistingDirectory=True)
    # Create path to directory containing the images that have traffic lights within them
    currentDirectory = os.getcwd()
    pathToImageDirectory = currentDirectory + ImagesWithTLDirectory
    os.chdir(pathToImageDirectory)
    imageIndex = 0
    # Prompt user for label
    print("Showing image of the video one frame at a time")
    print("If you see a traffic light in the image, press t")
    print("If you want to quit, press q")
    print("If there are no traffic lights in the image, press s\n")
    ## Read the input video capture one frame at a time until last frame is reached
    while trafficCapture.isOpened():
        # Read frame from video
        frameWasReturned, inputFrame = trafficCapture.read() 
        # If frame was successfully retrieved
        if frameWasReturned:
            cv.imshow("Traffic Image", inputFrame)
            inputFromUser = cv.waitKey(0)

            # If user noticed a traffic light in the picture
            if inputFromUser == ord('t'):
                imageFileName = "image" + str(imageIndex) + ".jpg"
                cv.imwrite(imageFileName, inputFrame)
                imageIndex += 1
            elif inputFromUser == ord('q'):
                exit(1)
            elif inputFromUser == ord('s'):
                continue
                
            # Go to next frame
        # No more video frames left
        else:
            break

    # Stop when the video is finished
    trafficCapture.release()


    # Close all windows
    cv.destroyAllWindows() 

