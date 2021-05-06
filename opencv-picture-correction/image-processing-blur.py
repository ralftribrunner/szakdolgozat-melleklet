import cv2 as cv
import numpy as np
from matplotlib import image, pyplot as plt
import os 

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images=load_images_from_folder("./")

rows, cols = (len(images), 3)
result = [[0 for i in range(cols)] for j in range(rows)]
for i in range(0,len(images)):
    result[i][0]=cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
    result[i][1]=cv.cvtColor(cv.GaussianBlur(images[i],(7,7),0), cv.COLOR_BGR2RGB)
    result[i][2]=cv.cvtColor(cv.blur(images[i],(5,5)), cv.COLOR_BGR2RGB)

j=1

for i in range(0,len(result)):
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][0]),plt.title('Original')
    j+=1
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][1]),plt.title('Gaussian blurred')
    j+=1
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][2]),plt.title('Averaged blurred')
    j+=1

plt.show()
