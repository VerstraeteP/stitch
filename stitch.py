import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

import math

import sys
from surface import predict_surface
from predict_renner import predict_renner


import matplotlib.pyplot as plt

def prepare_data_and_stitch(images,fps,scalingfactor):
	
	"""
	calculates the mask of the images
	:param images: set of finish images
	:parm scalingfactor: the amount of images that has to be used in the stitching
	:return: images,masks
	"""
	
	
	process_images=[]
	
	for i, data in enumerate(images):    				#go trought images and take scalingfactor into account and add it to new list
		if i % scalingfactor ==0:
			process_images.append(data)
	process_images.append(images[-1])				#add last frame of video to make sure the frame containing the finish line is in the image list
	
	del(images)							#delete old images list
	process_images.reverse()					#reverse framelist, so process_images[0]== finish frame, process_image[len(process_image)]== furthest frame from the finish line
	
	masks=predict_surface(process_images)				# call predict_surface function-> predict road surface
	
	stitchimage,transform,mask,totaltransform,teller,indexen,width,baselines=stitching(process_images,masks)	#call stitching function-> make stitching of finish frames
	process_images[:teller-1]											
	renners=predict_renner(process_images,masks)
	
	return stitchimage,transform,renners,scalingfactor,fps,mask,totaltransform,indexen,width,baselines,teller
	



def stitching(images,masks):
	total_transformation=0
	evaluate=0
	"""
	Stitch given images together to one uniform image
	:param images: set of images to be stitch
	:param masks: set of mask associated with the images
	:return: stitched image, Affinetransform
	"""
	cur_image=images[0]
	ttlchange=0
	ttlchangeteller=0
	detector = cv2.SIFT_create(contrastThreshold = 0.01)
	Affinetransformations=[[[1 , 0 ,500],[0,1,0]]]
	total_affine=[]
	base_msk= masks[0]
	cnt=0
	teller=1
	height, width = images[teller].shape[:2]
	curr = np.zeros((height*2,width*3, 3), np.uint8)
	baselines=[]
									#in the next paragraph some new image are created, with a specific height and width. In the comment: use of image
	
	base_gray=np.zeros((height*2,width*3, 3), np.uint8)
	total_mask=np.zeros((height*2,width*3), np.uint8)
	base_mask= np.zeros((height*2,width*3), np.uint8)
	mask_photo= np.zeros((height*2,width*3), np.uint8)
	increase=np.zeros((images[0].shape[0]*2,images[0].shape[1]*3,3), np.uint8)		#to increase the image 
	increasex=np.zeros((images[0].shape[0]*2,images[0].shape[1]*3,3), np.uint8)
	increase_mask=np.zeros((images[0].shape[0]*2,images[0].shape[1]*3), np.uint8)
	increase_mask_x=np.zeros((images[0].shape[0]*2,images[0].shape[1]*3), np.uint8)
	start_img=0
	base_gray[:images[0].shape[0],500:images[0].shape[1]+500]=cur_image
	total_mask[:base_msk.shape[0],500:base_msk.shape[1]+500]=base_msk
	heightc, widthc = curr.shape[:2]
	baseline=0
	baselinex=600
	largest=0
	times=0
	width=images[0].shape[1]
	baselineneg=600
	border=5
	vergroot=0
	indexen=[]
	lengte=len(images)
	for cur_image in images[1:]:								#go through all images starting from the second image
		try:										#try-catch block: it can occur that the stitching fails,if it fails:try to restore the maximum successful image
			if vergroot<20:								
				neg=False
				base_msk=masks[teller]
				base_msk[base_msk==0]=255

				base_msk[base_msk==1]=0
				base_mask[:,:]=0

				curr[:,:]=0	
				#curr[start_img:cur_image.shape[0]+start_img,:cur_image.shape[1]]=cur_image
				curr[300:cur_image.shape[0]+300,300:cur_image.shape[1]+300]=cur_image






				if cnt==0:							#
					base_mask[300+border:300+base_msk.shape[0]-border,300+border:300+base_msk.shape[1]-border]=base_msk[border:cur_image.shape[0]-border,border:cur_image.shape[1]-border]

					mask_photo[:base_msk.shape[0],500:500+base_msk.shape[1]]=base_msk
				else:											#if it isn't the first image: we will first using the total transformation of all the previous stitched images, to tranform current image, so the succes rate of keypoint detection increases

					#base_mask[start_img+border:base_msk.shape[0]-border+start_img,border:base_msk.shape[1]-border]=base_msk[border:cur_image.shape[0]-border,border:cur_image.shape[1]-border]

					base_mask[300+border:300+base_msk.shape[0]-border,300+border:300+base_msk.shape[1]-border]=base_msk[border:cur_image.shape[0]-border,border:cur_image.shape[1]-border]
					total_transformation=total_transformation[:2,:]					#transform 
					transformation=transformation[:2,:]
					total_transformation[0][2]=0
					total_transformation[1][2]=0
					tran=transformation.copy()
					tran[0][2]=0
					tran[1][2]=0

					base_mask = cv2.warpAffine(base_mask, total_transformation, (widthc, heightc)) #transform base_mask image ( mask corresponding curr image) with the total_transformation
					curr = cv2.warpAffine(curr, total_transformation, (widthc, heightc))		#transform curr image with the total transformation
					base_mask = cv2.warpAffine(base_mask, tran, (widthc, heightc))
					curr = cv2.warpAffine(curr, tran, (widthc, heightc))
					total_transformation = np.vstack((total_transformation,array))



					trans = np.vstack((tran,array))


					total_transformation=np.dot(total_transformation,trans)
					total_transformation=total_transformation[:2,:]
					total_affine.append(total_transformation)

					transformation=transformation[:2,:]
				#in the next paragraph: check if the result image needs to be enlarged, we take some threshold space needed to make sure the new image will fit in the total stitched image
				for k,i in enumerate(base_gray[baseline:,:]):					# check in y-direction, starting from baseline(last known position), go trough all lines
					if(~i.any()):								#if in any line all values are = 0 occures: this is the last empty line 
						baseline=k+baseline						#add number of line to baseline variable

						break

				transpose=base_gray[:baseline,baselinex+1:]					#new cropped image, cropped in x-direction from baselinex+1 to len(transpose[1]), cropped in y-direction from 0 to baseline

				tranposes=np.transpose(transpose,(1, 0, 2))					#transpose image matrix: [0,1,2],[3,4,5] -> [0,3],[1,4],[2,5]
				for k,i in enumerate(tranposes):						#loop trought transpose image, to find right edge of image
					if(~i.any()):								#find empty rows
						baselinex=k+baselinex

						break
				transpose=base_gray[:baseline,:baselineneg]					#new cropped image
				tranposes=np.transpose(transpose,(1, 0, 2))
				if cnt>0:
					for k,i in enumerate(tranposes):					#to find left edge of image
						if (i.any()):
							baselineneg=k
							break



				if (baselinex+int(cur_image.shape[1])/2)>base_gray.shape[1]:			#if detected edge + half of new image is smaller than current stitched image, than make stitched image larger
					vergroot+=1

					base_gray = np.append(base_gray,increase,axis=1)
					total_mask = np.append(total_mask,increase_mask,axis=1)
					mask_photo= np.append(mask_photo,increase_mask,axis=1)
					base_mask= np.append(base_mask,increase_mask,axis=1)
					curr = np.append(curr,increase,axis=1)
					increasex=np.zeros((images[0].shape[0],curr.shape[1],3), np.uint8)
					increase_mask_x=np.zeros((images[0].shape[0],curr.shape[1]), np.uint8)
					heightc, widthc = curr.shape[:2]


				if (baseline+cur_image.shape[0])>=base_gray.shape[0]:
					vergroot+=1
					mask_photo= np.append(mask_photo,increase_mask_x,axis=0)
					base_gray = np.append(base_gray,increasex,axis=0)
					base_mask= np.append(base_mask,increase_mask_x,axis=0)
					total_mask = np.append(total_mask,increase_mask_x,axis=0)
					curr = np.append(curr,increasex,axis=0)
					increase=np.zeros((curr.shape[0],images[0].shape[1],3), np.uint8)
					increase_mask=np.zeros((curr.shape[0],images[0].shape[1]), np.uint8)
					heightc, widthc = curr.shape[:2]

				if (baselineneg-cur_image.shape[1]/2)<0:
					vergroot+=1

					indexen.append(teller)
					baselines.append(baselineneg)
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
				
				



				base_features,base_descs=detector.detectAndCompute(base_gray,mask_photo)		#detect keypoints an their descriptors of image base_gray and use mask_photo as a mask
				next_features, next_descs = detector.detectAndCompute(curr,(base_mask))			#same as above, but for curr image and base_mask
				#cv2.imwrite("./drive/MyDrive/wkvideo/mask/"+str(teller)+".jpg",base_mask)		#write seperate masks to a folder


				"""
				bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
				matches = bf.match(base_descs,next_descs)
				matches = sorted(matches, key = lambda x:x.distance)
				filtered_matches=matches[:200]
				"""
				bf = cv2.BFMatcher()									#match the founded keypoints of the base_gray image with the keypoints of the curr image
				matches = bf.knnMatch(base_descs,next_descs,k=2)
				filtered_matches=[]
				for m,n in matches:
					if m.distance < 0.85*n.distance:						#only use the best ones
						filtered_matches.append(m)

				filtered_matches = np.asarray(filtered_matches)		


				src_pts  = np.float32([base_features[m.queryIdx].pt for m in filtered_matches]).reshape(-1,2)		#get the x, y coordinates from the best detected keypoints of the base_gray image
				dst_pts  = np.float32([next_features[m.trainIdx].pt for m in filtered_matches]).reshape(-1,2)		#get the x, y coordintates from the best detected keypoints of the curr image

				output = cv2.drawMatches(base_gray, base_features, curr, next_features, filtered_matches, None)		#function to draw the found matches
				cv2.imwrite("./drive/MyDrive/wkvideo/output/"+str(teller)+".jpg",output)

				

				transformation, status = cv2.estimateAffine2D(dst_pts, src_pts,ransacReprojThreshold=5,maxIters=10000 ,refineIters=1000)	#with the src_pts and dst_pts we can find a transformation between those points, ransacReprojThreshold= maximum reprojection error in the RANSAC algorithm to consider a point as an inlier.
				filtered_matche=[]
				array=np.array([0,0,1])

				Affinetransformations.append(transformation)

				mod_photo = cv2.warpAffine(curr, transformation, (widthc, heightc))					#apply transformation to the different images
				base_msk= cv2.warpAffine(base_msk, transformation, (widthc, heightc))	
				mask_photo = cv2.warpAffine(base_mask, transformation, (widthc, heightc))
				base_mask=cv2.warpAffine(base_mask, transformation, (widthc, heightc))

				(ret,data_map) = cv2.threshold(cv2.cvtColor(mod_photo, cv2.COLOR_BGR2GRAY),0, 255,cv2.THRESH_BINARY)	# make a mask in the new image

				contours, hierarchy = cv2.findContours(data_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)		#findcontour, otherwise, a black line in stitched image, we find contour and color it black, so mask is a little smaller
				contours1, hierarchy1 = cv2.findContours(base_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				cv2.drawContours(data_map, contours, -1, (0,255,255), 10)
				enlarged_base_img= cv2.bitwise_and(total_mask,total_mask, mask =np.bitwise_not(data_map))
				enlarged_base_img1 = cv2.bitwise_and(base_gray,base_gray,mask =np.bitwise_not(data_map))


				mod_photo= cv2.bitwise_and(mod_photo,mod_photo,mask =(data_map))
				mod_photo1= cv2.bitwise_and(base_msk,base_msk,mask =(base_msk))
				final_img = cv2.add(mod_photo,enlarged_base_img1,dtype=cv2.CV_8U)

				
				
					
				base_gray=final_img
																	# every 5 frames we save a temporary image, so if it fails we can go back
				if cnt==0:	
					prev_base_gray= base_gray
					prev_prev_base_gray=base_gray
					total_transformation=transformation
					total_transformation = np.vstack((total_transformation,array))
				elif cnt%5==0:
					prev_prev_base_gray= prev_base_gray
					prev_base_gray=base_gray






				cnt=cnt+1



				"""
				if cnt>0:
					for k, i in enumerate(data_map[:,:]):
							if(i.any()):
								start_img=k
								break

				"""
				total_mask= cv2.add(mod_photo1,enlarged_base_img,dtype=cv2.CV_8U)
				cv2.imwrite("./drive/MyDrive/wkvideo/base_gray/"+str(teller)+".jpg",base_gray)
				teller=teller+1
				
				
		except:												#if the stitching process fails: get max 5 frames back, and restore image
				
				ret= 5+cnt%5
				base_gray=prev_prev_base_gray							#get last image
				cv2.imwrite("endimage.jpg",prev_base_gray)
				teller-=ret
				Affinetransformations=Affinetransformations[:len(Affinetransformations)-ret]	#delete last ret frames of list of transformations
				if len(indexen)!=0:								#if 
					if(indexen[len(indexen)-1]>teller):
						indexen.pop()
						baselines.pop()

				
				break
				

	
		
	return base_gray,Affinetransformations,total_mask,total_affine,teller,indexen,width,baselines
			

		
		
