### This program is used to create a training set from a set 4 of folders storing 
### images of traffic lights with the three different colors and non-traffic
### light objects. The skeleton of the model is created from an InceptionV3 base
### model with additional layers built on top to achieve the traffic light color
### detection portion
import TrafficLightExtractor # our program containing the methods of producing images
                       # and detecting objects within them
import scipy # Import needed for training of model
import tensorflow as tf # library used for machine learning
from tensorflow import keras
import cv2 as cv # Library for image processing and computer vision
import numpy as np # library used for data processing and computation
import sys # Library used for interaction with python interpreter

# Import the InceptionV3 that will serve as the base model of our nerureal network
# and the preprocess_input function used to scale the input image pixels before they 
# are passed as input to the network
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
# Import Modelcheckpoint callback to tell the system when to save the weight file
from keras.callbacks import ModelCheckpoint
# Import Early Stopping point to tell the system when it can prematurely end the
# training session
from keras.callbacks import EarlyStopping
# Import the different layers that will be appended on top of the base model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.layers import Dropout, Dense, Flatten
# Import ImageDataGenerator such that we can create transformation on the dataset
# such that the model will be general and able to predict traffic lights
# when they appear in odd configurations
from keras.preprocessing.image import ImageDataGenerator
# Import loss function used to calculate the loss or the "error" in the predictions
# between the predicted label and the true label of the input images
from keras.losses import categorical_crossentropy
# Import Counter function to count the number of objects within each group
from collections import Counter

# Import the Sequential model type. We are only using one input into the 
# network, so we can use sequential model (one input, one output tensor)
# per layer
from keras.models import Sequential
# Import Adadelta gradient descent method to improve how well the model
# learns between layers
from keras.optimizers import Adadelta
from keras.utils import to_categorical # needed for binary categorization of labels


SHOW_NETWORK_ARCHITECTURE = False
# Map label descriptions to the corresponding integer labels of
# each group of traffic light
GREEN_LABEL = 0
YELLOW_LABEL = 1
RED_LABEL = 2
NON_TRAFFIC_LIGHT_LABEL = 3

# 0 = green
# 1 = yellow
# 2 = red
# 3 = not a traffic light
# Shuffles the set of images and their corresponding labels to incorporate more
# randomness into the training set
def shuffleTrainingSet(trainingImages, trainingLabels):
    # Randomly permute the values from 0 to the number of images in training set
    # to get a randomized ordering of indices
    randomizedIndexList = np.random.permutation(len(trainingImages))
    ## Create a list of randomized images and their corresponding labels
    shuffledImages = []
    shuffledLabels = []
    # For each of the indices in the random index list
    for i in randomizedIndexList:
        # Get the image and label corresponding to the random index and
        # append them to their respective lists
        shuffledImages.append(trainingImages[i])
        shuffledLabels.append(trainingLabels[i])
    
    # Return the shuffled training set
    return shuffledImages, shuffledLabels

### Builds the skeleton of a neural network using the InceptionV3 imagenet
### model as a base and replacing the top, classification layer of the model
### with the layers we defined
def buildModelArchitecture(numberOfClasses):

    # Take the imagenet-weighted version of the InceptionV3 model as the base of our 
    # neural network. We don't want to define our own top or classification layers
    # for our model, so we don't include the top
    inputHeight = 299
    inputWeight = 299
    numColorChannels = 3
    inputDimensions = (inputHeight, inputWeight, numColorChannels)
    baseModel = InceptionV3(weights='imagenet', include_top=False, input_shape=inputDimensions)

    # We only want to modify the weights of our classification layers and keep the weights of the
    # base imagenet model that same
    for layer in baseModel.layers:
        layer.trainable = False
    # Create a sequential model where each layer has one input tensor and one 
    # output tensor
    trafficLightColorModel = Sequential()

    # The bottom portion of the model will be the imagenet-weighted model
    trafficLightColorModel.add(baseModel)

    ## Add the additionaly layers on top of the imagenet-weighted models
    ## to perform the classifications that we need
    # First add layer to pool the spatial image data
    trafficLightColorModel.add(GlobalAveragePooling2D())
    # Add a dropout layer that drops input of random neurons to 0 during the 
    # training process. This helps to add some randomization to the model
    # that prevents the issue of overfitting
    trafficLightColorModel.add(Dropout(0.5))
    # Add a normal dense layer that outputs to 1024 nodes and uses the 
    # ReLu as activation function
    trafficLightColorModel.add(Dense(1024, activation='relu'))
    # Add layer to normalize the output. Keeps mean of output signal near
    # 0 and standard deviation near 1
    trafficLightColorModel.add(BatchNormalization())
    # Add a dropout layer that drops input of random neurons to 0 during the 
    # training process. This helps to add some randomization to the model
    # that prevents the issue of overfitting
    trafficLightColorModel.add(Dropout(0.5))
    # Add a normal dense layer that outputs to 512 nodes and uses the 
    # ReLu as activation function
    trafficLightColorModel.add(Dense(512, activation='relu'))
    trafficLightColorModel.add(Dropout(0.5))
    # Add a normal dense layer that outputs to 128 nodes and uses the 
    # ReLu as activation function
    trafficLightColorModel.add(Dense(128, activation='relu'))
    # For the output layer, use the softmax function for activation and
    # output to numberOfClasses output nodes
    trafficLightColorModel.add(Dense(numberOfClasses, activation='softmax'))

    if SHOW_NETWORK_ARCHITECTURE:
        # Display the base network architecture
        print('Layers: ', len(baseModel.layers))
        print("Shape:", baseModel.output_shape[1:])
        print("Shape:", baseModel.output_shape)
        print("Shape:", baseModel.outputs)
        baseModel.summary()

    # Return the newly-created architecture
    return trafficLightColorModel
if __name__ == "__main__":
    
    # Create an image data generator that will perform random operations on the images, 
    # including scaling, translations, rotations, etc. in order to ensure the model is 
    # generalized and able to handle any variations in traffic lights that it might later see
    datagen = ImageDataGenerator(rotation_range=5, width_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                                zoom_range=[0.7, 1.5], height_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                                horizontal_flip=True)
    desiredShape = (299, 299)

    ### Prepare the training set
    # Retrieve all of the images of traffic lights and store them in their own respective lists
    redTrafficLightFolder = "/red_traffic_lights"
    yellowTrafficLightFolder = "/yellow_traffic_lights"
    greenTrafficLightFolder = "/green_traffic_lights"
    nonTrafficLightFolder = "/not_traffic_lights"
    redTrafficLights = TrafficLightExtractor.getRGBImages(redTrafficLightFolder, desiredShape)
    yellowTrafficLights = TrafficLightExtractor.getRGBImages(yellowTrafficLightFolder, desiredShape)
    greenTrafficLights = TrafficLightExtractor.getRGBImages(greenTrafficLightFolder, desiredShape)
    notTrafficLights = TrafficLightExtractor.getRGBImages(nonTrafficLightFolder, desiredShape)

    # Create a list of labels for the images
    # Since the folders are already sorted, we can simply add integer labels 
    # deterministically for each folder
    trafficLightLabels = []
    # Add '0' for green traffic lights
    for i in range(len(greenTrafficLights)):
        trafficLightLabels.append(GREEN_LABEL)
    # Add '1' for yellow traffic lights
    for i in range(len(yellowTrafficLights)):
        trafficLightLabels.append(YELLOW_LABEL)
    # Add '2' for red traffic lights
    for i in range(len(redTrafficLights)):
        trafficLightLabels.append(RED_LABEL)
    # Add '3' for objects that aren't traffic lights or traffic lights with no
    # detectable color
    for i in range (len(notTrafficLights)):
        trafficLightLabels.append(NON_TRAFFIC_LIGHT_LABEL)



    # Count the number of labels occurring within each group and store this information
    # as a list of elements in the form (labelType, labelCount) 
    # where labelType is the type of the label and labelCount is the number of labels
    # within the set having that label
    labelCounts = Counter(trafficLightLabels)
    print("LabelCounts: ")
    print(labelCounts)
    # Show the user the total number of images in the training set and the distribution
    # of those images across the four groups
    print('Labels:', labelCounts)
    numberOfLabels = len(trafficLightLabels)
    print('Green: ', labelCounts[GREEN_LABEL])
    print('Yellow: ', labelCounts[YELLOW_LABEL])
    print('Red: ', labelCounts[RED_LABEL])
    print('Not Traffic Light: ', labelCounts[NON_TRAFFIC_LIGHT_LABEL])


    # Create a 2 dimensional numpy array that has four columns, one for
    # each label. A given row will have a 1 in the column that corresponds
    # to the correct label of the image
    trafficLightLabelsMatrix = np.ndarray(shape=(numberOfLabels, 4))
    # Set up a numpy array to store the individual pixel values of every single
    # image in the set
    imageMatrix= np.ndarray(shape=(numberOfLabels, desiredShape[0], desiredShape[1], 3))

    # Create a single list of all the image files
    allTrafficLightImages = []
    allTrafficLightImages.extend(greenTrafficLights)
    allTrafficLightImages.extend(yellowTrafficLights)
    allTrafficLightImages.extend(redTrafficLights)
    allTrafficLightImages.extend(notTrafficLights)

    # Exit the program with erorr if the number of images nad labels in the training set do not match
    if (len(allTrafficLightImages) != len(trafficLightLabels)):
        print("Number of training labels and images must match. Exiting program")
        exit(1)

    # First preprocess the images to ensure they have the [-1, 1] scale required by the neural network
    allTrafficLightImages = [preprocess_input(img) for img in allTrafficLightImages]
    # Shuffle the images in the training set to ensure consecutive images aren't the same
    (allTrafficLightImages, trafficLightLabels) = shuffleTrainingSet(allTrafficLightImages, trafficLightLabels)


    # Add the image and label data to the empty numpy arrays we previous created
    for i in range(len(trafficLightLabels)):
        # Store each label in the corresponding row of the numpy labels matrix
        trafficLightLabelsMatrix[i] = trafficLightLabels[i]
        # Store each image in the corresponding location within the empty Numpy image
        # matrix
        imageMatrix[i] = allTrafficLightImages[i]

    # Convert the list of labels to a matrix having a 1 in the column that corresponds to the objects 
    # identified label, and a 0 for all other columns (all other labels)
    for i in range(numberOfLabels):
        # We have four integer labels, representing the different colors of the 
        # traffic lights.
        trafficLightLabelsMatrix[i] = np.array(to_categorical(trafficLightLabels[i], 4))

    # Split the set of input images into two sets: one for the training set
    # and the other for the validation set. The training set is adjust the
    # individual weights of edges and the validation set is used to 
    # adjust hyperparameters
    dividingIndex= int(numberOfLabels * 0.75)
    trainingImages= imageMatrix[0:dividingIndex]
    validationImages = imageMatrix[dividingIndex:]
    validationLabels = trafficLightLabelsMatrix[dividingIndex:]
    trainingLabels = trafficLightLabelsMatrix[0:dividingIndex]


    # Generate a batch of randomly transformed images and their corresponding labels
    transformedTrainingData = datagen.flow(trainingImages, trainingLabels, batch_size=32)


    # Calculate weight of each class within the training set based upon the 
    weightsOfClasses = {GREEN_LABEL: numberOfLabels / labelCounts[0], YELLOW_LABEL: numberOfLabels / labelCounts[1],
            RED_LABEL: numberOfLabels / labelCounts[2], NON_TRAFFIC_LIGHT_LABEL: numberOfLabels / labelCounts[3]}
    print('Class weight:', weightsOfClasses)

    # Create a checkpoint object that will save a new checkpoint or version of the model's weight file every time
    # the cost function is improved between epochs
    modelCheckpoint = ModelCheckpoint("trafficLightColorModel2.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # Create an early stopping point for the model training such that the model will stop training if the cost
    # function of the model does not significantly improve for some consecutive number of epochs
    earlyStoppingPoint = EarlyStopping(min_delta=0.0005, patience=25, verbose=1)

    # Build the skeleton of the model with numberOfClasses Output nodes
    numberOfClasses = 4
    model = buildModelArchitecture(numberOfClasses)


    # Specify the training parameters for the model, including the method for
    # its gradient descent procedure used to reach more desirable weights,
    # learning rate, epsilon, etc
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(
        lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

    # Train the model for a set number of epochs on the transformed set of training
    # images nad used the validation image to 
    modelHistory = model.fit(transformedTrainingData, epochs=250, validation_data=(
        validationImages, validationLabels), shuffle=True, callbacks=[
        modelCheckpoint, earlyStoppingPoint], class_weight=weightsOfClasses)

