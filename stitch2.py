import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import time
import math
from cv2.xfeatures2d import matchGMS
from scc import ssc
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import sys
from surface import predict_surface
from predict_renner import predict_renner
from random import shuffle
from kdt import KDT_NMS
import matplotlib.pyplot as plt
def find_anomalies(data):
    #define a list to accumlate anomalies
    anomalies = []
    
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 2
    
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
   
   
   
    # Generate outliers
	


    for index,outlier in enumerate(data):
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(index)
    return anomalies
def prepare_data_and_stitch(images,fps,scalingfactor=3):
	
	"""
	calculates the mask of the images
	:param images: set of finish images
	:parm scalingfactor: the amount of images that has to be used in the stitching
	:return: images,masks
	"""
	
	process_images=[]
	
	for i, data in enumerate(images):
		if i % scalingfactor ==0:
			process_images.append(data)
	process_images.append(images[-1])
	print(len(process_images))
	process_images.reverse()
	fps_scaled=scalingfactor
	
	masks=predict_surface(process_images)
	
	stitchimage,transform,mask=stitching(process_images,masks)
	renners=predict_renner(process_images,masks)
	
	return stitchimage,transform,renners,fps_scaled,fps,mask
	



def stitching(images,masks):
	
	evaluate=0
	"""
	Stitch given images together to one uniform image
	:param images: set of images to be stitch
	:param masks: set of mask associated with the images
	:return: stitched image, Affinetransform
	"""
	ttlchange=0
	ttlchangeteller=0
	
	detector = cv2.ORB_create()
	Affinetransformations=[[[1 , 0 ,0],[0,1,0]]]
	
	base_msk= masks[0]
	cnt=0
	teller=1
	next_mask= masks[1]
	height, width = images[teller].shape[:2]
	curr = np.zeros((height*2,width*2, 3), np.uint8)
	base_gray=np.zeros((height*2,width*2, 3), np.uint8)
	total_mask=np.zeros((height*2,width*2), np.uint8)
	base_mask= np.zeros((height*2,width*2), np.uint8)
	mask_photo= np.zeros((height*2,width*2), np.uint8)
	increase=np.zeros((images[0].shape[0]*2,images[0].shape[1]*2,3), np.uint8)
	increasex=np.zeros((images[0].shape[0]*2,images[0].shape[1]*2,3), np.uint8)
	increase_mask=np.zeros((images[0].shape[0]*2,images[0].shape[1]*2), np.uint8)
	increase_mask_x=np.zeros((images[0].shape[0]*2,images[0].shape[1]*2), np.uint8)
	start_img=0
	base_gray[:images[0].shape[0],500:images[0].shape[1]+500]=images[0]
	total_mask[:base_msk.shape[0],500:base_msk.shape[1]+500]=base_msk
	heightc, widthc = curr.shape[:2]
	baseline=0
	baselinex=500
	largest=0
	times=0
	baselineneg=600
	border=5
	number_of_best=0
	


  
	for cur_image in images[1:]:
		times+=1
		best_transformation=[]
		
		best=None
		base_msk[base_msk==0]=255
		
		base_msk[base_msk==1]=0
		if cnt==0:
			mask_photo[:base_msk.shape[0],500:500+base_msk.shape[1]]=base_msk

		cnt=cnt+1
		cv2.imwrite("mask.jpg",mask_photo)
		cv2.imwrite("img.jpg",base_gray)
		


		
		base_features,base_descs=detector.detectAndCompute(base_gray,mask_photo)
		for k in range(number_of_best+1,number_of_best+3):
			print(number_of_best+k)
			cur_image=images[number_of_best+k]
			base_msk=masks[number_of_best+k]
			base_msk[base_msk==0]=255	
			base_msk[base_msk==1]=0
			base_mask[:,:]=0
			curr[:,:]=0	
			curr[start_img:cur_image.shape[0]+start_img,:cur_image.shape[1]]=cur_image
			base_mask[border:base_msk.shape[0]-border,border:base_msk.shape[1]-border]=base_msk[border:cur_image.shape[0]-border,border:cur_image.shape[1]-border]



			


			next_features, next_descs = detector.detectAndCompute(curr,(base_mask))


			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = bf.match(base_descs,next_descs)
			matches = sorted(matches, key = lambda x:x.distance)
			filtered_matches=matches[:400]
			
			output = cv2.drawMatches(base_gray, base_features, curr, next_features, filtered_matches, None)
			cv2.imwrite("output"+str(times)+".."+str(k)+".jpg",output)


			src_pts  = np.float32([base_features[m.queryIdx].pt for m in filtered_matches]).reshape(-1,2)
			dst_pts  = np.float32([next_features[m.trainIdx].pt for m in filtered_matches]).reshape(-1,2)
			


			transformation, status = cv2.estimateAffine2D(dst_pts, src_pts,ransacReprojThreshold=50,maxIters=10000 ,refineIters=10000)
			mod_photo = cv2.warpAffine(curr, transformation, (widthc, heightc))
			mask_photo = cv2.warpAffine(base_mask, transformation, (widthc, heightc))

			next_features, next_descs = detector.detectAndCompute(mod_photo,(mask_photo))
			matches = bf.match(base_descs,next_descs)
			matches = sorted(matches, key = lambda x:x.distance)
			filtered_matches=matches[:30]
			
			src_pts  = np.float32([base_features[m.queryIdx].pt for m in filtered_matches]).reshape(-1,2)
			dst_pts  = np.float32([next_features[m.trainIdx].pt for m in filtered_matches]).reshape(-1,2)
			dist=0
			for index,l in enumerate(src_pts):
				dist+=math.sqrt((src_pts[index][0]-dst_pts[index][0])**2+(src_pts[index][1]-dst_pts[index][1])**2)
			dist/=len(filtered_matches)
			if best==None or dist<best:
				best=dist
				best_transformation=transformation
				number_of_best=k
				best_mask=base_mask
				best_curr=curr.copy()
				best_msk =base_msk.copy()
           
      
      


		transformation=best_transformation
		print("best_transformation:")
		print(transformation)
		print("beste:"+str(number_of_best))
		cv2.imwrite("bestimage.jpg",best_curr)
		base_mask=best_mask
		base_msk=best_msk
		curr=best_curr
		Affinetransformations.append(transformation)
		
		mod_photo = cv2.warpAffine(best_curr, transformation, (widthc, heightc))
		base_msk = cv2.warpAffine(base_msk, transformation, (widthc, heightc))	
		mask_photo = cv2.warpAffine(base_mask, transformation, (widthc, heightc))
		base_mask=cv2.warpAffine(base_mask, transformation, (widthc, heightc))
		cv2.imwrite("over"+str(times)+".jpg",mod_photo)
		
	
		(ret,data_map) = cv2.threshold(cv2.cvtColor(mod_photo, cv2.COLOR_BGR2GRAY),0, 255,cv2.THRESH_BINARY)

		contours, hierarchy = cv2.findContours(data_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours1, hierarchy1 = cv2.findContours(base_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(data_map, contours, -1, (0,255,255), 10)

		enlarged_base_img1 = cv2.bitwise_and(base_gray,base_gray,mask =np.bitwise_not(data_map))
		
		mod_photo= cv2.bitwise_and(mod_photo,mod_photo,mask =(data_map))
		mod_photo1= cv2.bitwise_and(base_msk,base_msk,mask =(base_msk))
		final_img = cv2.add(mod_photo,enlarged_base_img1,dtype=cv2.CV_8U)
		
		base_gray=final_img
		cv2.imwrite("a"+str(times)+".jpg",base_gray)
		if cnt>0:
			for k, i in enumerate(data_map[:,:]):
					if(i.any()):
						start_img=k
						break
	

		
		teller=teller+1
		
	
		
		
	return base_gray,Affinetransformations,total_mask
			

		
		
