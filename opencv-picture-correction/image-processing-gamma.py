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

gamma= 3.5 #0.04-25.0

lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)


rows, cols = (len(images), 2)
result = [[0 for i in range(cols)] for j in range(rows)]
for i in range(0,len(images)):
    result[i][0]=cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
    gamma_res= cv.LUT(result[i][0], lookUpTable)
    new_image = cv.convertScaleAbs(gamma_res, alpha=1, beta=100)
    result[i][1]=gamma_res

j=1

for i in range(0,len(result)):
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][0]),plt.title('Original')
    j+=1
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][1]),plt.title('Modified')
    j+=1

plt.show()
