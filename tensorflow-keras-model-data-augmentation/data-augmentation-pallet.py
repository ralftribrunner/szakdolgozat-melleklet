# Augmentálást végez a raklapos képeken

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder,filename))
        if img is not None:
            x = img_to_array(img)
            x = x.reshape((1, ) + x.shape)
            images.append(x)

    return images

images=load_images_from_folder("./pallet")


datagen = ImageDataGenerator(
        vertical_flip = True,
		horizontal_flip = True,
        width_shift_range=0.1
        )
	
for i in range(0,len(images)):
	j = 0
	for batch in datagen.flow(images[i], batch_size = 1,
							save_to_dir ='generated-dataset/pallet',
							save_prefix ='pallet', save_format ='jpg'):
		j += 1
		if j >= 5:
			break

print("Data augmentation is done on the pallet dataset")

# Forrás: 
# https://www.geeksforgeeks.org/python-data-augmentation/
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

