#Importing the required packages
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import keras

from sklearn.preprocessing        import LabelBinarizer
from sklearn.model_selection      import train_test_split
from imutils                      import paths

from keras.models                 import Sequential

from keras.layers.convolutional   import Conv2D
from keras.layers.convolutional   import MaxPooling2D
from keras.layers.convolutional   import ZeroPadding2D

from keras.layers.core            import Activation
from keras.layers.core            import Flatten
from keras.layers.core            import Dropout
from keras.layers.core            import Dense

from keras                        import backend as K
from keras.utils                  import to_categorical
from keras.preprocessing.image    import img_to_array
from keras.preprocessing.image    import ImageDataGenerator
from keras.optimizers             import SGD


#Processing the malginant and benign images
class LoadingSkinCancerImages:
    def __init__(self, width, height):
        #Initializing image dimensions
        self.width  = width
        self.height = height

    def dataProcessing(self):
        #Creating lists to store features and targets
        data   = []
        labels = []

        #Locating paths to the images
        imageLocs = list(paths.list_images("Skin Cancer Dataset"))
        print("|SKINCAN NN| Loading in the images...")

        #Loading in and processing individual images
        for(counter, imageLoc) in enumerate(imageLocs):
            #Grabbing the skin images (labels = Malignant/Benign)
            image = cv2.imread(imageLoc)
            label = imageLoc.split(os.path.sep)[-2]

            #Processing images to standard dimensions
            image = cv2.resize(image, (self.width, self.height), interpolation = cv2.INTER_AREA)
            image = img_to_array(image, data_format = None)

            data.append(image)
            labels.append(label)

            #Updating image load-in progress
            if counter%100 == 0:
                print("|SKINCAN NN| Loaded in image {}/{}".format(counter, len(imageLocs)))

        #Returning a tuple of images and class labels
        print("|SKINCAN NN| Finished loading-in images...")
        return (np.array(data), np.array(labels))

#Initializing image properties
width  = 64
height = 64
depth  = 3
shape  = (width, height, depth)

(data, labels) = LoadingSkinCancerImages(width, height).dataProcessing()

#Normalizing pixel values to be between 0 and 1
data = data.astype("float")/255.0

#Splitting the training and test data
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
    test_size = 0.2, random_state = 42)

#Setting labels to integer values
lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY  = lb.fit_transform(testY)

#Creating the network
print("|SKINCAN NN| Creating the network...")
model = Sequential()

#Checking data format in the keras backend
if K.image_data_format() == "channels_first":
    shape = (depth, height, width)

#Building the layers of the model

epochs = 100

print("|SKINCAN NN| Compiling the network...")
model.add(ZeroPadding2D((1,1), input_shape = shape))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

opt = SGD(lr = 0.01)

model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

#Training the model
print("|SKINCAN NN| Training the network...")
M = model.fit(trainX, trainY, validation_data = (testX, testY), 
    batch_size = 16, epochs = epochs)

#Saving the model
model.save("skin_cancer_nn.h5")

#Plotting loss and accuracy in training
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, epochs), M.history["loss"],         label = "Training Loss")
plt.plot(np.arange(0, epochs), M.history["val_loss"],     label = "Loss")
plt.plot(np.arange(0, epochs), M.history["accuracy"],     label = "Training Accuracy")
plt.plot(np.arange(0, epochs), M.history["val_accuracy"], label = "Accuracy")

plt.title("Loss and Accuracy during the 'Training' process ")
plt.xlabel("Epoch Number")
plt.ylabel("Loss and Accuracy")
plt.legend()
plt.show()

plt.savefig("Training Loss and Accuracy.png")

