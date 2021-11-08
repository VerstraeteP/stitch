import matplotlib.pyplot as plt
import cv2
import numpy as np
def backgroundsubtraction(images):
	
	medianFrame = np.median(images, axis=0).astype(dtype=np.uint8)   				#medianfilter on image
	medianFrame= cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)					#change color of frame to grayscale
	medianFrame[medianFrame<200]=0									#use threshold if pixelvalue is less than 200 set to zero
	
		
	canny_output = cv2.Canny(medianFrame, 100,255)
	contours, dino= cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)        # find contour of white area, 
	
	for cnt in contours:
        	x,y,w,h=cv2.boundingRect(cnt)								#fill contour with white pixels and take some extra pixels
        	cv2.rectangle(medianFrame,(x,y),(x+w+10,y+h+10),(255,255,255),-1)
         
	
	(thresh, blackAndWhiteImage) = cv2.threshold(medianFrame, 127, 255, cv2.THRESH_BINARY)		#
	
	blackAndWhiteImage[blackAndWhiteImage>0]=255
	
	return blackAndWhiteImage
