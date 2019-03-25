import numpy as np 
#import cv2 as cv
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from skimage import exposure

#This portion deals with extracting the code from the kaggle dataset
f = open(r'shipsnet.json')
dataset = json.load(f)
f.close()

data = np.array(dataset['data']).astype('uint8')
img_length = 80

#Bit of reshaping to get the dataset in order
print(np.shape(data))
data = data.reshape(-1,3,img_length,img_length).transpose([0,2,3,1])
print(np.shape(data))

#Convert the images to grayscale
data_gray = [ color.rgb2gray(i) for i in data]
print(np.shape(data_gray))

#pixels per cell
ppc = 16

hog_images = []
hog_features = []
i =0 
for image in data_gray:

    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    # Uncomment this part to see the images and their HOGs 
    """
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    #print(hog_image)
    i = i + 1
    if i>10: #change the condition to view more images
        break
    """
    hog_images.append(hog_image)
    hog_features.append(fd)


#plt.imshow(np.array(hog_images))

#Extract the labels from the dataset
labels = np.array(dataset['labels']).reshape(len(dataset['labels']),1)

#Create the classifier
clf = svm.SVC()

hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features,labels))

#Shuffle the data set
np.random.shuffle(data_frame)

percentage = 80

partition = int(len(hog_features)*percentage/100)

x_train = data_frame[:partition,:-1]
x_test  = data_frame[partition:,:-1]

y_train = data_frame[:partition,-1:].ravel() 
y_test = data_frame[partition:,-1:].ravel()

#Train the classifier with the training data
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))



