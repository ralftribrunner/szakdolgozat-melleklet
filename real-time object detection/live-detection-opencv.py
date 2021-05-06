import cv2
import numpy as np
import jetson.inference
import jetson.utils


def cuda_to_cv(cuda_image):
	# rgb_img = jetson.utils.loadImage(cuda_image)
	rgb_img = cuda_image
	bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width,
							    height=rgb_img.height,
							    format='bgr8')
	jetson.utils.cudaConvertColor(rgb_img, bgr_img)
	jetson.utils.cudaDeviceSynchronize()
	return jetson.utils.cudaToNumpy(bgr_img)

def cv_to_cuda(cv_image):
	bgr_img = jetson.utils.cudaFromNumpy(cv_image, isBGR=True)
	rgb_img = jetson.utils.cudaAllocMapped(width=bgr_img.width,
							    height=bgr_img.height,
							    format='rgb8') #gray32f-vel
								#nem volt jó, mert a glDisplay, csak rgb-t és rgba-t támogat
	jetson.utils.cudaConvertColor(bgr_img, rgb_img)
	return rgb_img

net= jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera=jetson.utils.gstCamera(1280,720, "/dev/video0")
display=jetson.utils.glDisplay()

while display.IsOpen():
	img, width, height = camera.CaptureRGBA()
	img=cuda_to_cv(img)
	blur = cv2.blur(img,(2,2))
	#blur = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)	
	blur= cv_to_cuda(blur)
	detections=net.Detect(blur,width,height)
	display.RenderOnce(blur,width, height)
	display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
	
#https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-to-cv.py
#https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py
