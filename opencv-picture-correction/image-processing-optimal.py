# A tapasztaltak alapján a legjobbnak vélt képkorrekció.
# A beolvasott képeken először a kontraszt és fényerőt változtatja meg, majd gaussi homályosítást alkalmaz.

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

alpha=0.7 #0.0-3.0 default:1 opt:0.7
beta=50   #0-200   default:0 opt:50    
#A kontraszttal (alpha) a nagy szín különbségét tudtam csökkenteni
#A fényerővel (beta) visszavilágostottam hogy az eredetihez hasonló legyen

rows, cols = (len(images), 2)
result = [[0 for i in range(cols)] for j in range(rows)]
for i in range(0,len(images)):
    result[i][0]=cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
    new_image = np.zeros(images[i].shape, images[i].dtype)
    image=images[i]
    new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta) # kontraszt és fényerő megváltoztatása az alpha és beta értékével
    result[i][1]=cv.GaussianBlur(new_image,(7,7),0)
    result[i][1]=cv.cvtColor(new_image, cv.COLOR_BGR2RGB)

j=1

for i in range(0,len(result)):
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][0]),plt.title('Original')
    j+=1
    plt.subplot(len(result),len(result[i]),j),plt.imshow(result[i][1]),plt.title('Modified')
    j+=1

plt.show()

# forrás:
# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder/30230738
# https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
# https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html