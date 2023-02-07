import TrafficLightExtractor
import cv2 as cv
import tensorflow as tf


if __name__ == "__main__":

    # Set desired dimensions of output frames
    desiredHeight = 1920/1.4
    desiredWidth = 1080/1.4

    # Set the desired dimensions of the p
    objectDetectionModel = TrafficLightExtractor.loadCOCONetworkModel()
    # Load the traffic color detection model
    trafficColorModel = tf.keras.models.load_model("trafficLightColorModel.h5")
    inputFileName ='DetectionVideos/DrivingThroughSeattle.mp4'
    
    # Get file name without the file extension
    fileName = inputFileName.split(".mp4")[0]
    print(fileName)


    # Load the traffic video
    trafficCapture = cv.VideoCapture(inputFileName)

    # Prompt the user to either skip the frame or detect objects within the
    # the frame
    print("\n\nDetecting object in traffic images!")
    print("Please press the s key to skip to the next frame")
    print("Alternatively, press the d key to detect objects in the frame ")
    print("If you would like to quit at any point, press the q key")

    # Process the video
    trafficLightCounter = 0
    frameCounter = 0
    while trafficCapture.isOpened():
    # Capture one frame at a time
        success, frame = trafficCapture.read() 

        # Do we have a video frame? If true, proceed.
        if success:
            # First show user frame, and let them decide whether to detect 
            # objects in it or skip to next frame
            cv.imshow("Input Image", frame)
            userInput = cv.waitKey(0)
            # Skip to the next frame if user enters s
            if userInput == ord('s'):
                continue
            # If user enters d, then detect objects in the fame, label them
            # and show the resulting output frame to the user
            if userInput == ord('d'):
                # Resize the frame
                width = int(desiredHeight)
                height = int(desiredWidth)
                frame = cv.resize(frame, (width, height))

                # Store the original frame
                original_frame = frame.copy()
                outputFrame, predictionResults = TrafficLightExtractor.detObjVidFrame(frame, objectDetectionModel, 
                                                labelImage = True, TLDetectModel= trafficColorModel)
                # outputFrame, predictionResults = detectObjectsInVideoFrame(frame, objectDetectionModel, 
                #                                                             labelImage=True)
                outputFrame = cv.cvtColor(outputFrame, cv.COLOR_BGR2RGB)
                cv.imshow("Labeled image", outputFrame)
                cv.waitKey(0)

            elif userInput == ord('q'):
                exit(1)
        # Otherwise, if the frame was not read properly
        else:
            break

    # Stop when the video is finished
    trafficCapture.release()

    # Close all windows
    cv.destroyAllWindows() 




