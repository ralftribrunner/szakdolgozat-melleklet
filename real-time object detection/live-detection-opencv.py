# Valósidejű, kamerás OpenCV-s homályosítást végez a Jetson Nano-n 

import cv2
import numpy as np
import jetson.inference
import jetson.utils

# cuda kép cv formátumba konvertálása
def cuda_to_cv(cuda_image):
	# rgb_img = jetson.utils.loadImage(cuda_image)
	rgb_img = cuda_image
	# bgr formátumú cuda memória allokálás
	bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width,
							    height=rgb_img.height,
							    format='bgr8')
	jetson.utils.cudaConvertColor(rgb_img, bgr_img) # színkonverzió
	jetson.utils.cudaDeviceSynchronize() # meg kell várni, hogy teljesen lefusson a művelet
	return jetson.utils.cudaToNumpy(bgr_img) # cuda memóriából numoy tömböt készít

# cv kép cuda formátumba konvertálása
def cv_to_cuda(cv_image):
	bgr_img = jetson.utils.cudaFromNumpy(cv_image, isBGR=True) # cuda kép konverziója BGR képből
	rgb_img = jetson.utils.cudaAllocMapped(width=bgr_img.width,
							    height=bgr_img.height,
							    format='rgb8') #gray32f-vel
								#nem volt jó, mert a glDisplay, csak rgb-t és rgba-t támogat
	jetson.utils.cudaConvertColor(bgr_img, rgb_img)
	return rgb_img

net= jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5) # feltanított neurális hálózati modell
camera=jetson.utils.gstCamera(1280,720, "/dev/video0") # kamera elérése
display=jetson.utils.glDisplay() # új ablak (software window)

while display.IsOpen(): # amíg az ablak nyitva van 
	img, width, height = camera.CaptureRGBA() # pillanatkép készítése
	img=cuda_to_cv(img) 
	blur = cv2.blur(img,(2,2)) # homályosítás 2×2-es kernellel
	#blur = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)	 # éldetekciót végez, ezt kihagytam, mivel csak kísérleti jelleggel próbáltam ki
	blur= cv_to_cuda(blur)
	detections=net.Detect(blur,width,height) # visszatért a képpel, rajta a detektált objektumokkal
	display.RenderOnce(blur,width, height) # megjeleníti a képet
	display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

# Forrás:
# https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-to-cv.py
# https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py
# https://www.youtube.com/watch?v=bcM5AQSAzUY
