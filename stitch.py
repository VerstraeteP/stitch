import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import time
import math
import sys
from demo import predict_surface
from predict_renner import predict_renner
def prepare_data_and_stitch(images,fps,scalingfactor=2):
	
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
	
	base_gray[:images[0].shape[0],500:images[0].shape[1]+500]=images[0]
	total_mask[:base_msk.shape[0],500:base_msk.shape[1]+500]=base_msk
	heightc, widthc = curr.shape[:2]
	baseline=0
	baselinex=500
	largest=0
	times=0
	baselineneg=600
	
	for cur_image in images[1:]:
		neg=False
		if len(base_msk.shape)==3:
			base_msk=  cv2.cvtColor(base_msk, cv2.COLOR_BGR2GRAY)
		
		base_mask[:,:]=0
		curr[:,:]=0	
		base_mask[:base_msk.shape[0],:base_msk.shape[1]]=(base_msk)
		curr[:cur_image.shape[0],:cur_image.shape[1]]=cur_image
		if cnt == 0:
			mask_photo[:base_msk.shape[0],500:500+base_msk.shape[1]]=base_msk
			cnt=cnt+1	
		
		
		base_msk[base_msk<255]=0
		
		starttime=time.time()
		
		
		for k,i in enumerate(base_gray[baseline:,:]):
			if(~i.any()):
				baseline=k+baseline
				break
		
		transpose=base_gray[:baseline,baselinex+1:]
		tranposes=np.transpose(transpose,(1, 0, 2))
		for k,i in enumerate(tranposes):
			if(~i.any()):
				baselinex=k+baselinex
				break
		transpose=base_gray[:baseline,:baselineneg]
		tranposes=np.transpose(transpose,(1, 0, 2))
		for k,i in enumerate(tranposes):
			if (i.any()):
				baselineneg=k
				break
			
		
		if (baselinex+int(cur_image.shape[1])/2)>base_gray.shape[1]:
			print("increasing")
			base_gray = np.append(base_gray,increase,axis=1)
			total_mask = np.append(total_mask,increase_mask,axis=1)
			mask_photo= np.append(mask_photo,increase_mask,axis=1)
			base_mask= np.append(base_mask,increase_mask,axis=1)
			curr = np.append(curr,increase,axis=1)
			increasex=np.zeros((images[0].shape[0],curr.shape[1],3), np.uint8)
			increase_mask_x=np.zeros((images[0].shape[0],curr.shape[1]), np.uint8)
			heightc, widthc = curr.shape[:2]
			
	
		if (baseline+cur_image.shape[0])>=base_gray.shape[0]:
			mask_photo= np.append(mask_photo,increase_mask_x,axis=0)
			base_gray = np.append(base_gray,increasex,axis=0)
			base_mask= np.append(base_mask,increase_mask_x,axis=0)
			total_mask = np.append(total_mask,increase_mask_x,axis=0)
			curr = np.append(curr,increasex,axis=0)
			increase=np.zeros((curr.shape[0],images[0].shape[1],3), np.uint8)
			increase_mask=np.zeros((curr.shape[0],images[0].shape[1]), np.uint8)
			heightc, widthc = curr.shape[:2]
		
		if (baselineneg-cur_image.shape[1]/2)<0:
			print("neg")
			base_gray = np.append(base_gray,increase,axis=1)
			total_mask = np.append(total_mask,increase_mask,axis=1)
			mask_photo= np.append(mask_photo,increase_mask,axis=1)
			base_mask= np.append(base_mask,increase_mask,axis=1)
			curr = np.append(curr,increase,axis=1)
			base_gray[:,baselineneg:]=base_gray[:,:base_gray.shape[1]-baselineneg]
			curr[:,baselineneg:]=curr[:,:curr.shape[1]-baselineneg]
			total_mask[:,baselineneg:]=total_mask[:,:total_mask.shape[1]-baselineneg]
			base_mask[:,baselineneg:]=base_mask[:,:base_mask.shape[1]-baselineneg]
			mask_photo[:,baselineneg:]=mask_photo[:,:mask_photo.shape[1]-baselineneg]
			base_gray[:,:baselineneg]=0
			total_mask[:,:baselineneg]=0
			base_mask[:,:baselineneg]=0
			mask_photo[:,:baselineneg]=0
			curr[:,:baselineneg]=0
			increasex=np.zeros((images[0].shape[0],curr.shape[1],3), np.uint8)
			increase_mask_x=np.zeros((images[0].shape[0],curr.shape[1]), np.uint8)
			heightc, widthc = curr.shape[:2]
			baselineneg+=increase.shape[1]
			neg=True
		
		mask_photo[mask_photo<255]=0
		
		times+=1
		print(times)
		"""
		if times>60:
			plt.imshow(base_gray)
			plt.figure(1)
			plt.imshow(mask_photo)
			plt.figure(2)
			plt.imshow(curr)
			plt.figure(3)
			plt.imshow(base_mask)
			plt.figure(4)
			plt.imshow(img3)
			plt.show()
		"""
		base_features, base_descs = detector.detectAndCompute(base_gray,mask_photo)
		
		next_features, next_descs = detector.detectAndCompute(curr,(base_mask))	
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(base_descs,next_descs)
		matches = sorted(matches, key = lambda x:x.distance)
		filtered_matches=matches[:100]
		
		img3 = cv2.drawMatches(base_gray,base_features,cur_image,next_features,matches[:200],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		
		
		
		src_pts  = np.float32([base_features[m.queryIdx].pt for m in filtered_matches]).reshape(-1,1,2)
		dst_pts  = np.float32([next_features[m.trainIdx].pt for m in filtered_matches]).reshape(-1,1,2)
		transformation, status = cv2.estimateAffine2D(dst_pts, src_pts)
		Affinetransformations.append(transformation)
		mod_photo = cv2.warpAffine(curr, transformation, (widthc, heightc))
		mask_photo = cv2.warpAffine(base_mask, transformation, (widthc, heightc))
		base_msk = cv2.warpAffine(base_msk, transformation, (widthc, heightc))
		
		ttldistance=0
		tellers=0
		if evaluate==1:
			base_features, base_descs = detector.detectAndCompute(base_gray,mask_photo)
			next_features, next_descs = detector.detectAndCompute(mod_photo,(base_msk))
				
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = bf.match(base_descs,next_descs)
			matches = sorted(matches, key = lambda x:x.distance)
			filtered_matches=matches[:200]
				
			src_pts  = np.float32([base_features[m.queryIdx].pt for m in filtered_matches]).reshape(-1,1,2)
			dst_pts  = np.float32([next_features[m.trainIdx].pt for m in filtered_matches]).reshape(-1,1,2)
			for k in range(len(src_pts)):
				distance = math.sqrt((src_pts[k][0][0]-dst_pts[k][0][0])**2+(src_pts[k][0][1]-dst_pts[k][0][1])**2)
					
				if distance!=0 and distance<10:
						
					tellers+=1
					ttldistance+=distance
			if tellers!=0:
					
				ttlchange+=ttldistance/tellers
				ttlchangeteller+=1
		(ret,data_map) = cv2.threshold(cv2.cvtColor(mod_photo, cv2.COLOR_BGR2GRAY),0, 255,cv2.THRESH_BINARY)

		contours, hierarchy = cv2.findContours(data_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours1, hierarchy1 = cv2.findContours(base_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(data_map, contours, -1, (0,255,255), 10)
		enlarged_base_img= cv2.bitwise_and(total_mask,total_mask, mask =np.bitwise_not(data_map))
		enlarged_base_img1 = cv2.bitwise_and(base_gray,base_gray,mask =np.bitwise_not(data_map))
		mod_photo= cv2.bitwise_and(mod_photo,mod_photo,mask =(data_map))
		mod_photo1= cv2.bitwise_and(base_msk,base_msk,mask =(base_msk))
		final_img = cv2.add(mod_photo,enlarged_base_img1,dtype=cv2.CV_8U)
		"""
		if neg==True:
			plt.imshow(enlarged_base_img1)
			plt.figure(2)
			plt.imshow(mod_photo)
			plt.show()
		"""
		total_mask= cv2.add(mod_photo1,enlarged_base_img,dtype=cv2.CV_8U)
		
		base_gray=final_img
	
		base_msk=masks[teller]
		
		teller=teller+1
		
	
		
		
	return base_gray,Affinetransformations,total_mask
			

		
		
