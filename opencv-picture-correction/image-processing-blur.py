# A szkript gaussi, illetve átlagoló homályosítást végez a beolvasott képeken

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

# A képeket a módosításokkal együtt egy összes_képek_száma × 3 méretű táblázatba fogom helyezni
# Egy sorban ugyanaz az alapkép van. 
# Az első oszlop az eredeti képet, a második az átlagoló homályosított képet, 
# a harmadik pedig a gauss-al homályosított képet tartalmazza.
# Az OpenCV BGR színformátumot használ, így ezt vissza kellett konvertálni RGB-be,
# hogy emberi szemmel látható legyen a változás.
rows, cols = (len(images), 3)
result = [[0 for i in range(cols)] for j in range(rows)]
for i in range(0,len(images)):
    result[i][0]=cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
    # A homályosító műveleteknél a második paraméter a kernel nagysága
    result[i][1]=cv.cvtColor(cv.GaussianBlur(images[i],(7,7),0), cv.COLOR_BGR2RGB)
    result[i][2]=cv.cvtColor(cv.blur(images[i],(5,5)), cv.COLOR_BGR2RGB)

j=1

#Az eredmények megjelenítése
for i in range(0,len(result)):
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][0]),plt.title('Original')
    j+=1
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][1]),plt.title('Gaussian blurred')
    j+=1
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][2]),plt.title('Averaged blurred')
    j+=1

plt.show()

# forrás:
# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder/30230738
# https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html