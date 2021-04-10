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
def prepare_data_and_stitch(images,fps,scalingfactor=10):
	
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
	total_transformation=0
	evaluate=0
	"""
	Stitch given images together to one uniform image
	:param images: set of images to be stitch
	:param masks: set of mask associated with the images
	:return: stitched image, Affinetransform
	"""
	ttlchange=0
	ttlchangeteller=0
	
	detector = cv2.SIFT_create()
	Affinetransformations=[[1 , 0 ,0],[0,1,0]]
	
	base_msk= masks[2]
	cnt=0
	teller=3
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
	base_gray[:images[0].shape[0],500:images[0].shape[1]+500]=images[2]
	total_mask[:base_msk.shape[0],500:base_msk.shape[1]+500]=base_msk
	heightc, widthc = curr.shape[:2]
	baseline=0
	baselinex=500
	largest=0
	times=0
	baselineneg=600
	border=5
	for cur_image in images[3:]:
		cv2.imwrite("1.png",images[1])
		cv2.imwrite(str(times)+".png",masks[teller])
		neg=False
		base_msk=masks[teller]
		base_msk[base_msk==0]=255
		print(total_transformation)
		base_msk[base_msk==1]=0
		base_mask[:,:]=0
		
		curr[:,:]=0	
		#curr[start_img:cur_image.shape[0]+start_img,:cur_image.shape[1]]=cur_image
		curr[:cur_image.shape[0],:cur_image.shape[1]]=cur_image
		
		

	
		"""
		if len(base_msk.shape)==3:
			base_msk=  cv2.cvtColor(base_msk, cv2.COLOR_BGR2GRAY)
		"""
	
		if cnt==0:
			base_mask[border:base_msk.shape[0]-border,border:base_msk.shape[1]-border]=base_msk[border:cur_image.shape[0]-border,border:cur_image.shape[1]-border]

			mask_photo[:base_msk.shape[0],500:500+base_msk.shape[1]]=base_msk
		else:
			
			#base_mask[start_img+border:base_msk.shape[0]-border+start_img,border:base_msk.shape[1]-border]=base_msk[border:cur_image.shape[0]-border,border:cur_image.shape[1]-border]
			
			base_mask[border:base_msk.shape[0]-border,border:base_msk.shape[1]-border]=base_msk[border:cur_image.shape[0]-border,border:cur_image.shape[1]-border]
			for k in Affinetransformations[1:]:
				
				base_mask = cv2.warpAffine(base_mask, k, (widthc, heightc))
				curr = cv2.warpAffine(curr, k, (widthc, heightc))

			
		
		cv2.imwrite("aftermask"+str(times)+".jpg",base_mask)
		
		
		cv2.imwrite("mask.jpg",base_msk)
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
		if cnt>0:
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
		
		

		#base_features, base_descs = detector.detectAndCompute(base_gray,mask_photo)
		
			
		base_features,base_descs=detector.detectAndCompute(base_gray,mask_photo)
		#base_features=goodFeaturesToTrack(base_gray, mask=mask_photo,minDistance=10)
		#base_features,base_descs=detector.compute(base_gray,base_features)
		#next_features=goodFeaturesToTrack(curr, mask=base_mask,minDistance=10)
		#next_features,base_descs=detector.compute(curr,next_features)
		next_features, next_descs = detector.detectAndCompute(curr,(base_mask))
		
		"""
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(base_descs,next_descs)
		matches = sorted(matches, key = lambda x:x.distance)
		filtered_matches=matches[:200]
		"""
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(base_descs,next_descs,k=2)
		filtered_matches=[]
		for m,n in matches:
    			if m.distance < 0.8*n.distance:
        			filtered_matches.append(m)
		
		filtered_matches = np.asarray(filtered_matches)		
		"""
		
		data=[]
		good_matches=[]
		for k in filtered_matches:
			src_pts  = np.float32(base_features[k.queryIdx].pt).reshape(-1,2)
			dst_pts  = np.float32(next_features[k.trainIdx].pt).reshape(-1,2)
			
			if src_pts[0][1]>dst_pts[0][1]:
				data.append(src_pts[0][1]-dst_pts[0][1])
				good_matches.append(k)
		data=np.array(data)
		d = np.abs(data - np.median(data))
		mdev = np.median(d)
		s = d/mdev if mdev else 0.
		for index,k in enumerate(s):
			if k>2:
				good_matches[index]=0
		good_matches = [x for x in good_matches if x !=0]
		
		base_features=[base_features[m.queryIdx] for m in filtered_matches]
		base_descs=[base_descs[m.queryIdx] for m in filtered_matches]
		#base_descs=np.array(base_descs)
		#base_features=np.array(base_features)
		
		 

		next_features=[next_features[m.trainIdx] for m in filtered_matches]
		next_descs=[next_descs[m.trainIdx] for m in filtered_matches]
		
	
		base_descs=np.array(base_descs)
		next_descs=np.array(next_descs)
		
		img3 = cv2.drawKeypoints(base_gray, base_features,base_gray, color=(255, 0, 0))
    

		      
		
		base_features,base_descs = KDT_NMS(base_features, base_descs, r=30, k_max=200)
		#base_descs=base_descs.astype('uint8')
		#base_features = ssc(base_features, 100, 0.1, base_gray.shape[1], base_gray.shape[0])
		#base_features, base_descs= detector.compute(base_gray,base_features)
		base_descs=np.array(base_descs)
		next_descs=np.array(next_descs)
		
		
		base_descs=base_descs.astype('uint8')
		next_descs=next_descs.astype('uint8')
	
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		filter_matches = bf.match(base_descs,next_descs)
		copy=base_gray.copy()
		img3e = cv2.drawKeypoints(copy, base_features,copy, color=(255, 0, 0))
		cv2.imwrite("before.jpg",img3)
		cv2.imwrite("after.jpg",img3e)
		print(len(filtered_matches))
		"""
		
		src_pts  = np.float32([base_features[m.queryIdx].pt for m in filtered_matches]).reshape(-1,2)
		dst_pts  = np.float32([next_features[m.trainIdx].pt for m in filtered_matches]).reshape(-1,2)
		#base_featur=[base_features[m.queryIdx] for m in filtered_matches]
		#next_featur=[next_features[m.trainIdx] for m in filtered_matches]
		"""
		model, inliers = ransac((src_pts, dst_pts),AffineTransform, min_samples=40,residual_threshold=8, max_trials=10000)
		n_inliers = np.sum(inliers)
		inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
		inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
		placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
		image3 = cv2.drawMatches(base_gray, inlier_keypoints_left, cur_image, inlier_keypoints_right, placeholder_matches, None)
		
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
		matches_all = matcher.match(base_descs, next_descs)
		matches_all=matches_all[:200]
		start = time.time()
		matches_gms = matchGMS(base_gray.shape[:2], curr.shape[:2], base_features, next_features, matches_all, withScale=False, withRotation=False, thresholdFactor=6)
		end = time.time()
		print('Found', len(matches_gms), 'matches')
		print('GMS takes', end-start, 'seconds')
		output = cv2.drawMatches(base_gray, base_features, curr, next_features, matches_gms, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		src_pts = np.float32([ base_features[m.queryIdx].pt for m in matches_gms ]).reshape(-1, 2)
		dst_pts = np.float32([ next_features[m.trainIdx].pt for m in matches_gms ]).reshape(-1, 2)
		"""
		output = cv2.drawMatches(base_gray, base_features, curr, next_features, filtered_matches, None)
		cv2.imwrite("before"+str(times)+".jpg",output)

		
		#transformation, status = cv2.estimateAffine2D(dst_pts, src_pts)
		transformation, status = cv2.estimateAffine2D(dst_pts, src_pts,ransacReprojThreshold=2,maxIters=10000 ,refineIters=10000)
		#base_features=[]
		#next_features=[]
		filtered_matche=[]
		array=np.array([0,0,1])
		

		for index,k in enumerate(status):
				if k==1:
					#base_features.append(base_featur[index])
					#next_features.append(next_featur[index])
					filtered_matche.append(filtered_matches[index])
		#base_features=np.array(base_features)
		#next_features=np.array(next_features)
		filtered_matche=np.array(filtered_matche)
		

		count=0
		for k in status:
			if k==1:
				count+=1
		print("before:"+str(src_pts.shape))
		print("matches:"+ str(count))
		Affinetransformations.append(transformation)
		
		mod_photo_temp = cv2.warpAffine(curr, transformation, (widthc, heightc))
		base_msk_temp= cv2.warpAffine(base_msk, transformation, (widthc, heightc))	
		mask_photo_temp = cv2.warpAffine(base_mask, transformation, (widthc, heightc))
		base_mask_temp=cv2.warpAffine(base_mask, transformation, (widthc, heightc))
		
		next_features, next_descs = detector.detectAndCompute(mod_photo_temp,(mask_photo_temp))
		matches = bf.match(base_descs,next_descs)
		matches = sorted(matches, key = lambda x:x.distance)
		filtered_matches=matches[:200]
		src_pts  = np.float32([base_features[m.queryIdx].pt for m in filtered_matches]).reshape(-1,1,2)
		dst_pts  = np.float32([next_features[m.trainIdx].pt for m in filtered_matches]).reshape(-1,1,2)
		distance=0
		for k in range(len(src_pts)):
			distance += math.sqrt((src_pts[k][0][0]-dst_pts[k][0][0])**2+(src_pts[k][0][1]-dst_pts[k][0][1])**2)
		distance/=k
		print(distance)
		
		mod_photo=mod_photo_temp 
		base_msk=base_msk_temp	
		mask_photo=mask_photo_temp 
		base_mask=base_mask_temp
		(ret,data_map) = cv2.threshold(cv2.cvtColor(mod_photo, cv2.COLOR_BGR2GRAY),0, 255,cv2.THRESH_BINARY)

		contours, hierarchy = cv2.findContours(data_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours1, hierarchy1 = cv2.findContours(base_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(data_map, contours, -1, (0,255,255), 10)
		#enlarged_base_img= cv2.bitwise_and(total_mask,total_mask, mask =np.bitwise_not(data_map))
		enlarged_base_img1 = cv2.bitwise_and(base_gray,base_gray,mask =np.bitwise_not(data_map))
		
		
		mod_photo= cv2.bitwise_and(mod_photo,mod_photo,mask =(data_map))
		mod_photo1= cv2.bitwise_and(base_msk,base_msk,mask =(base_msk))
		final_img = cv2.add(mod_photo,enlarged_base_img1,dtype=cv2.CV_8U)
		
		#total_mask= cv2.add(mod_photo1,enlarged_base_img,dtype=cv2.CV_8U)
		
		base_gray=final_img
		
		print(total_transformation)
		if cnt==0:	
		
			total_transformation=transformation
			total_transformation = np.vstack((total_transformation,array))
		else:
			
			
			total_transformation = np.vstack((total_transformation,array))
			print(total_transformation)
			transformation = np.vstack((transformation,array))

			print(total_transformation.shape)
			print(transformation.shape)
			total_transformation=np.dot(total_transformation,transformation)
		total_transformation=total_transformation[:2,:]
		transformation=transformation[:2,:]
		print(transformation)
		
		
		cnt=cnt+1
		
			

		
		
		
		"""
		for z in range(4):
			
			
			



			
		
			next_features, next_descs = detector.detectAndCompute(mod_photo,(base_mask))
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = bf.match(base_descs,next_descs)
			matches = sorted(matches, key = lambda x:x.distance)
			filtered_matches=matches[:200]
			
			src_pts=[]
			dst_pts=[]
			src1=[]
			dst1=[]
			distance=[]
			sum=0
			src_pts  = np.float32([base_features[m.queryIdx].pt for m in filtered_matches]).reshape(-1,2)
			dst_pts  = np.float32([next_features[m.trainIdx].pt for m in filtered_matches]).reshape(-1,2)
			base_feat=[base_features[m.queryIdx] for m in filtered_matches]
			for index,k in enumerate(src_pts):
				dist=math.sqrt((src_pts[index][0]-dst_pts[index][0])**2+(src_pts[index][1]-dst_pts[index][1])**2)
				sum+=dist
				
				
				distance.append(dist)
				
				    
				src1.append(src_pts[index])
				dst1.append(dst_pts[index])
			
			indexd=find_anomalies(distance)
			
			print(sum)
			for m in indexd:
				
				
				src1[m]=[None,None]
				dst1[m]=[None,None]
				
				ind=base_features.index(base_feat[m])
				
				del(base_features[ind])
				
				
				
				base_descs=np.delete(base_descs,ind,axis=0)
				
				
				distance[m]=20212
			src2=[]
			dst2=[]
			
			for index,k in enumerate(src1):
				if k[0]==None and k[1]==None:
					pass
					
				else:
					src2.append(src1[index])
					dst2.append(dst1[index])
			
			distance = [x for x in distance if x !=20212]
			matches = bf.match(base_descs,next_descs)
			matches = sorted(matches, key = lambda x:x.distance)
			
			filtered_matches=matches[:maxindex]
			
			
					
			src1=np.array(src2)
			dst1=np.array(dst2)
			
		
			src1.astype(np.float32)
			dst1.astype(np.float32)
			
			
			transformation, status = cv2.estimateAffine2D(dst1, src1,ransacReprojThreshold=50,maxIters=10000 ,refineIters=10000)

			mod_photo = cv2.warpAffine(mod_photo, transformation, (widthc, heightc),flags=flag)
			base_msk = cv2.warpAffine(base_msk, transformation, (widthc, heightc),flags=flag)
			mask_photo = cv2.warpAffine(mask_photo, transformation, (widthc, heightc),flags=flag)
			base_mask=cv2.warpAffine(base_mask, transformation, (widthc, heightc),flags=flag)
			

		
		img3 = cv2.drawMatches(base_gray,base_features,mod_photo,next_features,filtered_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)		
		cv2.imwrite("image"+str(times)+".jpg",img3)


		
		
		ttldistance=0
		tellers=0
		if evaluate==1:
			base_features, base_descs = detector.detectAndCompute(base_gray,mask_photo)
			next_features, next_descs = detector.detectAndCompute(mod_photo,(base_msk))
				
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = bf.match(base_descs,next_descs)
			matches = sorted(matches, key = lambda x:x.distance)
			filtered_matches=matches[:20]
				
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
		"""
		
		if cnt>0:
			for k, i in enumerate(data_map[:,:]):
					if(i.any()):
						start_img=k
						break
	

		
		teller=teller+1
		
	
		
		
	return base_gray,Affinetransformations,total_mask
			

		
		
