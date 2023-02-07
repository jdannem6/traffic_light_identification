### This program defines the functions used to detect the traffic lights, 
### traffic light color, and moving objects within given pictures 

## Program uses a neural network that has been pretrained to detect a number of 
## of objects, including cars, buses, and traffic lights. This network was trained on
# the Common Objects in Context (COCO) dataset 


import numpy as np # library for data processing and computation
import cv2 as cv # computer vision library
import tensorflow as tf# Library for machine learning methods
from tensorflow import keras #  Neural network API library
import glob # Library used to retrieve all files in given directories
import os # Library for operating system operations
import shutil # library for deleting path

# preprocess_input necessary to esentially normalize images before they
# can be processed with inception_v3 model
from keras.applications.inception_v3 import preprocess_input

# Create constants to store the integer labels of each object to be detected
# where the integer label is that defined for that object in the given neural
# network model
PERSON_LABEL = 1
TRAFFIC_LIGHT_LABEL = 10
CAR_LABEL = 3
TRUCK_LABEL = 6
BUS_LABEL = 6

CONFIDENCE_THRESHOLD = 0.3 # Predictions with confidences less than this threshold 
                           # are disregarded


# Returns a list of files stored within a given directory. Used to retrieve all 
# the paths of testing or training images
def getFilesInDirectory(directoryName):
    # List of all files in directory
    files = []
    # Create absolute path to the given directory
    directoryToPath = os.getcwd() + directoryName 
    # Find all files that match the file pattern to list of files
    filePattern = directoryToPath + '/*'

    for file in glob.iglob(filePattern, recursive=True):
        files.append(file)
    return files

# Creates a new directory, deleting any matching pre-existing directory if
# instructed to do so
def createDirectory(directoryName, deleteExistingDirectory=True):
    # Create directory if it doesn't already exist to store the traffic light file
    currentDirectory = os.getcwd()
    pathToDirectory = currentDirectory + directoryName
    # Delete the directory if deleteExistingDirectory is True
    if deleteExistingDirectory is True and os.path.exists(pathToDirectory):
        # If directory has files in it, delete using shutil utility function
        if len(os.listdir(pathToDirectory)) != 0:
            shutil.rmtree(pathToDirectory) # remove folder and contents
        # otherwise if it has no files, just remove the directory
        else: 
            os.rmdir(pathToDirectory)
    # Create new directory to store the traffic lights
    os.mkdir(pathToDirectory)



# Downloads a pretrained object detection model from tensorflow server given the name of 
# the model
def loadModelFromServer(modelName):
    # Create the URL that will be used to find the specific tensorflow model
    skeletonOfURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    fileExt = '.tar.gz'
    modelURL= skeletonOfURL + modelName + fileExt

    # Download tar file storing the object detection model, decompress the file, and 
    # store the model in directoryOfModel
    directoryOfModel = tf.keras.utils.get_file(modelName, untar=True, origin=modelURL)

    print("Model was saved at: " + str(directoryOfModel))

    # Define the export directory of the model suchcle that it can be loaded 
    modelExportDirectory = str(directoryOfModel) + "/saved_model"
    # Load model from export directory
    downloadedModel = tf.saved_model.load(export_dir=modelExportDirectory)

    return downloadedModel


# Loads a pretrained neural network model that was trained using COCO dataset
def loadCOCONetworkModel():
    modelName = "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
    # Load the given model from TensorFlow Server
    model = loadModelFromServer(modelName)
    return model

# Retrieves all image files within a given directory and converts to RGB format
def getRGBImages(directoryOfImages, desiredShape = None):
    # Retrieve the file names
    imageFiles = getFilesInDirectory(directoryOfImages)
    # Read in each image and convert from the openCV BGR format
    # to RGB format
    RGBImages = []
    for fileName in imageFiles:
        # read in the image
        image = cv.imread(fileName)
        # Convert to RGB format
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # If user specified a desired dimensions for the frames, 
        # change to those dimensions
        if desiredShape is not None:
            image = cv.resize(image, desiredShape)
        RGBImages.append(image)
    return RGBImages

# Determines whether a given prediction is considered a duplicate of previously
# encountered predictions. Any predicitons that are positioned less than
# the SIMILARITY_THRESHOLD away from one another are considered duplicates
def isDuplicatePrediction(boundingBoxes, indexOfPrediction, similarity_threshold):
    currentBox = boundingBoxes[indexOfPrediction]

    # Computer center of current prediction's bounding box
    xCenterCurrent = (currentBox[1] + currentBox[3])/2
    yCenterCurrent  = (currentBox[0] + currentBox[2])/2

    # Compare center of current bounding box to center of all previous predictions
    for i in range(indexOfPrediction):
        # Calculate center of other bounding box
        otherBox = boundingBoxes[i]
        xCenterOther = (otherBox[1] + otherBox[3])/2
        yCenterOther = (otherBox[0] + otherBox[2])/2


        # If difference between the centers is less than similarity threshold, 
        # current prediction is a duplicate
        if (abs(xCenterCurrent - xCenterOther) < similarity_threshold):
            if (abs(yCenterCurrent- yCenterOther) < similarity_threshold):
                return True
    # If no duplicates, return false
    return False


## Detects all traffic lights within an image and saves them to a file
def detectTrafficLights(inputImage, objectDetectionModel, confidenceThreshold):
    # Convert frame from opencv's native  BGR format to RGB format
    inputImage = cv.cvtColor(inputImage, cv.COLOR_BGR2RGB)
    # cv.imshow("image", inputImage)
    # cv.waitKey(0)
    # Convert the input image to a tensor (this change its dimensions to 
    # (height, width, numberOfColorChannels) 
    inputTensor = tf.convert_to_tensor(inputImage)
    # Add another dimension at beginning of tensor to denote the number
    # of samples in the tensor (in this it is 1)
    inputTensor = inputTensor[tf.newaxis, ...]
    # Generate set of predictions for the objects within the input image
    predictionResults = objectDetectionModel(inputTensor)


    # Convert tensors to numPy array such that we can index and process the 
    # prediction results
    numDetections = int(predictionResults.pop('num_detections'))
    for key, value in predictionResults.items():
        attributeTable = value[0, : numDetections].numpy()
        predictionResults[key] = attributeTable
    predictionResults['numDetections'] = numDetections

    ## Extract all prediction information about traffic lights in the images
    trafficLightDetections = {}
    # Create lists to store all information about traffic lights
    trafficLightClasses = []
    trafficLightBoxes = []
    trafficLightScores = [] # Confidences of traffic light predictions
    # For each object detected in image
    # print(predictionResults['detection_classes'])
    # print(predictionResults['detection_scores'])
    for i in range(len(predictionResults['detection_classes'])):
        # If the detected object is a traffic light
        if predictionResults['detection_classes'][i] == TRAFFIC_LIGHT_LABEL:
            # If the confidence of the prediction is greater than the confidence threshold
            if predictionResults['detection_scores'][i] > confidenceThreshold:
                # Add the traffic light's detection information
                # Store its object class as an integer
                trafficLightClasses.append(int(predictionResults['detection_classes'][i]))
                # Store the information about its bounding box
                trafficLightBoxes.append(predictionResults['detection_boxes'][i])
                # Store the confidence level of prediction
                trafficLightScores.append(predictionResults['detection_scores'][i])

    # Store this traffic light detection information together
    trafficLightDetections['detection_classes'] = trafficLightClasses
    trafficLightDetections['detection_boxes'] = trafficLightBoxes
    trafficLightDetections['detection_scores'] = trafficLightScores

    return trafficLightDetections

# Takes a directory of images which each contain traffic lights and an object 
# detection model and uses the model to extract all the images of traffic 
# lights and saves them in trafficLightDirectory, a common directory storing  
# all traffic lights
def extractTrafficLights(imageDirectory, objectDetectionModel, trafficLightDirectory):
    # Get list of all images that contain traffic lights
    imagesWithTrafficLights = getFilesInDirectory(imageDirectory)
    ## Create the traffic light directory to store images of traffic lights within
    createDirectory(trafficLightDirectory)
    # Change to traffic light directory
    currentPath = os.getcwd()
    trafficLightDirectory = currentPath + trafficLightDirectory
    os.chdir(trafficLightDirectory)
    # Define desired size to make frames before they are processed
    desiredHeight = 1920
    desiredWidth = 1080
    desiredResolution = (desiredHeight, desiredWidth)

    trafficLightCounter = 0
    firstFrame = True
    ## Read each image from the image directory, one at a time
    for imagePath in imagesWithTrafficLights:
        # Read the image 
        inputImage = cv.imread(imagePath)

        # Get width and height of frames
        if firstFrame:
            heightOfFrame = inputImage.shape[0]
            widthOfFrame = inputImage.shape[1]
            firstFrame = False

        # # Resize the frame to desired resolution
        resizedFrame = cv.resize(inputImage, desiredResolution)
        # cv.imshow('frame', resizedFrame)
        # cv.waitKey(0)
        print(imagePath)
        # Search for traffic lights in frame
        trafficLightDetections = detectTrafficLights(resizedFrame, objectDetectionModel,
                                                        confidenceThreshold=0.4)
        print(trafficLightDetections)
        # For each detected traffic light in the frame
        for i in range(len(trafficLightDetections['detection_boxes'])):
            ### Extract the traffic light from the image
            # Get bounding box of detected object
            boundingBox = trafficLightDetections["detection_boxes"][i]

            ## Isolate the traffic light in image
            # Get the x,y coordinates of the corners of the bounding
            # box and convert them from fractional form to pixels
            yMin= int(boundingBox[0] * heightOfFrame)
            xMin = int(boundingBox[1] * widthOfFrame)
            # x-y coordinates of bottom right corner of traffic light box
            yMax = int(boundingBox[2] * heightOfFrame)
            xMax =  int(boundingBox[3] * widthOfFrame)

            # Store the new bounding box dimension information
            boundingBox[0] = yMin
            boundingBox[1] = xMin
            boundingBox[2] = yMax
            boundingBox[3] = xMax

            # Create an image containing only the traffic light
            trafficLightImage = inputImage[yMin:yMax, xMin:xMax]
            # Write to the traffic light image to the directory for later processing
            # traffic light folder for later processing 
            fileName = "trafficLight" + str(trafficLightCounter) + ".jpg"
            cv.imwrite(fileName, trafficLightImage)
            cv.resize(trafficLightImage, (1000, 1000))

            # Track the number of  traffic lights detected
            trafficLightCounter += 1

    # Close all frame windows that are open
    cv.destroyAllWindows() 

# Takes a frame of an RGB video and the corresponding prediction information, writes 
# labels for all traffic lights colors and moving objects
def labelFrame(frame, predictionsOnImage, trafficColorDetectionModel = None):

    # For each prediction made upon the image
    for i in range(len(predictionsOnImage['boxes'])):
        # Get bounding box of detected object
        boundingBox = predictionsOnImage["boxes"][i]
        # Get integer class of prediction
        objectClass = predictionsOnImage["detection_classes"][i]
        # Get confidence of prediction
        predConfidence = predictionsOnImage["detection_scores"][i]
        # truncate the prediction confidence value
        predConfidence = round(predConfidence, 2)
        boundingBoxColor = None
        predictionLabel = ""

        # Color of bounding box corresponds to confidence of prediction
        Red = (0, 0, 255)
        Yellow = (0, 255, 255)
        Green = (0, 255, 0)
        if predConfidence > 0.85:
            boundingBoxColor = Green
        elif predConfidence >= 0.50:
            boundingBoxColor = Yellow
        elif predConfidence < 0.50:
            boundingBoxColor = Red

        # Get the label corresponding to the integer class
        if objectClass == PERSON_LABEL:
            predictionLabel = "Person " + str(predConfidence)
        elif objectClass == CAR_LABEL:
            predictionLabel = "Car " + str(predConfidence)
        elif objectClass == BUS_LABEL:
            predictionLabel = "Bus " + str(predConfidence)
        elif objectClass == TRUCK_LABEL:
            predictionLabel = "Truck " + str(predConfidence)
        elif objectClass == TRAFFIC_LIGHT_LABEL:
            predictionLabel = "Traffic Light " + str(predConfidence)
            
            ## Model believes many objects are traffic lights
            ## Only label a traffic light if the confidence is high
            if predConfidence > 0.4:

                # Isolate traffic light in image
                # x-y coordinates of top left corner of traffic light box
                xCorner1 = boundingBox["x1"]
                yCorner1 = boundingBox["y1"]
                # x-y coordinates of bottom right corner of traffic light box
                xCorner2 = boundingBox["x2"]
                yCorner2 = boundingBox["y2"]

                # Create an image containing only the traffic light
                trafficLightImage = frame[yCorner1:yCorner2, xCorner1:xCorner2]

                # Recolor traffic light image to openCV's BGR format and write to
                # traffic light folder for later processing 
                recoloredTrafficLightImage = cv.cvtColor(trafficLightImage, cv.COLOR_RGB2BGR)

                # Resize image to dimensions used for predictions
                resizedTrafficLight = cv.resize(trafficLightImage, (299, 299))

                ## Form prediction on color of traffic light
                # image must first be preprocessed to match the dimensions of the neural network
                # and converted to numpy array
                resizedTrafficLight = np.array([preprocess_input(resizedTrafficLight)])
                # Make prediction
                prediction = trafficColorDetectionModel.predict(resizedTrafficLight)
                label = np.argmax(prediction)
                predConfidence = round(np.max(prediction),2)
                # Update color of bounding box based on prediction of traffi
                if predConfidence > 0.85:
                    boundingBoxColor = Green
                elif predConfidence >= 0.50:
                    boundingBoxColor = Yellow
                elif predConfidence < 0.50:
                    boundingBoxColor = Red

                if label == 0:
                    predictionLabel = "Green " + str(predConfidence)
                elif label == 2: 
                    predictionLabel = "Red " + str(predConfidence)
                # Otherwise, if color of the light could not be detected, do not label the traffic light
                else:
                    continue
            # If there is not much confidence that it is a traffic light, go to the next prediction
            else:
                continue

        # If the color and label of the prediction were successfully assigned and this prediction is 
        # not a duplicate of previous predictions, label the prediction on the original image
        duplicatePrediction = isDuplicatePrediction(predictionsOnImage['detection_boxes'], 
                                                    i, similarity_threshold = 0.005)
        if (duplicatePrediction is False and boundingBoxColor is not None 
            and predictionLabel and predConfidence > CONFIDENCE_THRESHOLD):
            # Label object
            cv.putText(frame, predictionLabel, (boundingBox["x1"], boundingBox["y1"]), 
                         cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=2)
            # Draw bounding rectangle over object
            cv.rectangle(frame, (boundingBox["x1"], boundingBox["y1"]), (boundingBox["x2"], boundingBox["y2"]), 
                         boundingBoxColor, thickness=3)
    # Write the changes to the labeled image file
    labeledFrame  = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    return labeledFrame
       
def detObjVidFrame(inputframe, objmodel,labelImage = False, TLDetectModel = None):

    # The frame will be in BGR format, so wee convert to RGB
    # So we convert frame from BGR to RGB format
    #This is possible with CV Library 
    inputimage = cv.cvtColor(inputframe, cv.COLOR_BGR2RGB)

    #We need to have the frame with type as Tensor
    #So we convert the input image to a tensor
    #This changes its dimensions to height, width, numberOfColorChannels
    inputtensor = tf.convert_to_tensor(inputimage)

    # We add another dimension which specifies the sample size of frames
    # Which in this case its 1
    #We add the dimension at front of tensor
    inputtensor = inputtensor[tf.newaxis, ...]

    # Generate predictions on input tensor
    # This generatse 
    predictionres = objmodel(inputtensor)

    #We get many detections in the above variable
    #But we will take only what's necessary to us
    print("\n\n\n")
    print('Detection classes:', predictionres['detection_classes'])
    print('Detection Boxes:', predictionres['detection_boxes'])
    print('Detection scores: ', predictionres['detection_scores'])


    #Now we convert these tensor to numpy arrays
    # To process the results that we received earlier
    numDetections = int(predictionres.pop('num_detections'))
    for key, value in predictionres.items():
        attributeTable = value[0, : numDetections].numpy()
        predictionres[key] = attributeTable
    predictionres['numDetections'] = numDetections


    #Now we map the detection classes to integers
    predictionres['detection_classes'] = predictionres['detection_classes'].astype(np.int64)

    # Show what objects were detected
    # print('\n\n\n')
    # for i in range(len(predictionres['detection_classes'])):
    #     print("Object class: " + str(predictionres['detection_classes'])[i])
    #     print("Confidence of prediction: " + str(predictionres['detection_scores'][i]))
    #Now we change the dimensions of the box to pixel based presentation
    #We also store the height and width of the frame in pixels
    height = inputframe.shape[0]
    width = inputframe.shape[1]

    #Here we declare a list of boxes which stores new dimensions
    Boxes = []

    #Writing a loop to get new dimensions for all the boxes
    for box in predictionres['detection_boxes']:

        #Declaring a dictionary which stores the dimensions of a box
        Box = {}

        #Here we store the fractional values of corners of the box
        xmin = box[1]
        ymin = box[0]
        xmax = box[3]
        ymax = box[2]
        

        #Here we calculate the same corner values of the box in pixels
        xMinPixels = int(xmin * width)
        yMinPixels = int(ymin * height)
        xMaxPixels = int(xmax * width)
        yMaxPixels = int(ymax * height)

        #We store the new dimenions in its respective box
        Box["x1"] = xMinPixels
        Box["y1"] = yMinPixels
        Box["x2"] = xMaxPixels
        Box["y2"] = yMaxPixels

        # Add this bounding box to list 
        Boxes.append(Box)

    # Update the boxes attribute of the predictions with the new pixel-based boundingBox dimensions
    predictionres["boxes"] = Boxes


    # checking label image is true
    #If yes return the labeled frame too else only return prediction results
    if labelImage:
        labeledFrame = labelFrame(inputframe, predictionres, TLDetectModel)
        return labeledFrame, predictionres
    else:
        return predictionres


if __name__ == "__main__":
    # Set desired dimensions of output frames
    desiredHeight = 1920
    desiredWeight = 1080
    # Load the predefined object detection model
    objectDetectionModel = loadCOCONetworkModel()

    ### Extract all traffic lights from the video into traffic light directory
    inputImageDirectory = "/ImagesWithTrafficLights"
    trafficLightDirectory = "/Traffic_Lights2"
    extractTrafficLights(inputImageDirectory, objectDetectionModel, trafficLightDirectory)





