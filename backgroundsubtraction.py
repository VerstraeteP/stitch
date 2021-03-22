import matplotlib.pyplot as plt
import cv2
import numpy as np
def backgroundsubtraction(images):
	
	medianFrame = np.median(images, axis=0).astype(dtype=np.uint8)   
	medianFrame= cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
	
	medianFrame[medianFrame<200]=0
	
	
	canny_output = cv2.Canny(medianFrame, 100,255)
	contours, dino= cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	for cnt in contours:
        	x,y,w,h=cv2.boundingRect(cnt)
        	cv2.rectangle(medianFrame,(x,y),(x+w+10,y+h+10),(255,255,255),-1)
         
	
	(thresh, blackAndWhiteImage) = cv2.threshold(medianFrame, 127, 255, cv2.THRESH_BINARY)
	
	blackAndWhiteImage[blackAndWhiteImage>0]=255
	
	return blackAndWhiteImage
