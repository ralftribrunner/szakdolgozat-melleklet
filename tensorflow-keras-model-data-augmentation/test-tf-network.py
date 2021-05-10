# Képek átfuttatása a beolvasott neurális hálózati modellen és detekciós idő mérése

import numpy as np
import tensorflow as tf
import pathlib
import time
from tensorflow import keras

# Elmentett modell beolvasása
model=keras.models.load_model('tf-pallet-detection.h5')
img_height = 500
img_width = 15
class_names= ["empty", "pallet"]
data_dir = pathlib.Path('./test')
test_pallet = list(data_dir.glob('pallet/*'))
test_empty = list(data_dir.glob('empty/*'))
pallet_correct=0
empty_correct=0

# Klasszifikáció az összes tesztképre
for i in range(0,len(test_pallet)):
    print("________________________________________________________________")
    print("Image: "+str(i+1)+".")
    preprocess_start=time.time()

    img = keras.preprocessing.image.load_img(
        test_pallet[i], target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    preprocess_finish=time.time()
    preprocess_time=(preprocess_finish-preprocess_start) 

    prediction_start=time.time()
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    prediction_finish=time.time()
    prediction_time=(prediction_finish-prediction_start) 
    total_time=(preprocess_time+prediction_time) 

    print("----------------------------------------------------------------")
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence. It should a pallet."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    #Preprocessing time: kép beolvasása
    print("Preprocessing time: " + str(preprocess_time*1000)  +" ms")
    #Prediction time: kép átfuttatása a neurális hálózaton
    print("Prediction time: " + str(prediction_time*1000) + " ms")
    #Teljes detektálási idő
    print("TOTAL: " + str(total_time*1000) + " ms ")

    if(class_names[np.argmax(score)]=="pallet"):
        pallet_correct+=1
    else:
         print("WRONG DETECTION")
    preprocess_start=time.time()

    img = keras.preprocessing.image.load_img(
        test_empty[i], target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    preprocess_finish=time.time()
    preprocess_time=(preprocess_finish-preprocess_start) 

    prediction_start=time.time()

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    prediction_finish=time.time()
    prediction_time=(prediction_finish-prediction_start) 
    total_time=(preprocess_time+prediction_time) 

    print("----------------------------------------------------------------")
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence. It should be empty."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    print("Preprocessing time: " + str(preprocess_time*1000)  +" ms")
    print("Prediction time: " + str(prediction_time*1000) + " ms")
    print("TOTAL: " + str(total_time*1000) + " ms ")

    if(class_names[np.argmax(score)]=="empty"):
        empty_correct+=1
    else:
         print("WRONG DETECTION")
print("________________________________________________________________")
# Összes raklapos képből hány lett helyes
print("Pallet Accuracy: " + str(pallet_correct) + "/30")
# Összes raklap nélküli képből hány lett helyes
print("Empty Accuracy: " + str(empty_correct) + "/30")

# Forrás:
# https://www.tensorflow.org/guide/keras/save_and_serialize#introduction
# https://www.tensorflow.org/tutorials/images/classification

