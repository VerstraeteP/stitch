from imutils.video import FileVideoStream
import time
import argparse
import math
from stitch import prepare_data_and_stitch
from backgroundsubtraction import backgroundsubtraction
import matplotlib.pyplot as plt
from  calculate_positions import calculate_pos
from imutils.video import FPS
import json
import cv2
import sys

def start(video):
	
	"""
	Process a video given by "arg.video", predicts if frame is a finish frame
	"""
	"""
	parser = argparse.ArgumentParser(description='Video-based sprint lane analysis to improve safety in road bicycle racing')
	#parser.add_argument('--video', default="vidbocht.mp4", help='video file to process')
	parser.add_argument('--renners', default=30, help='total renners')
	args = parser.parse_args()
	"""
	
	frame_list,fps=process_video(video)				#process video 
	
	scalingfactor=math.ceil(len(frame_list)/600)			#to avoid memory problems in stitching: only use the first 600 frames( first 20sec, most videos 30fps)
	"""
	if scalingfactor==0:
		scalingfactor=1
	"""
	if scalingfactor == 0 : scalingfactor = 1			#if len(frame_list)<0: set scalingfactor to 1

	stitched_image,affinetransformation,renners,fps_scaled,fps,mask,total_transform,indexen,width,baseline,teller=prepare_data_and_stitch(frame_list,fps,scalingfactor)
	stitched=stitched_image.copy() 
	solution,renner,transformaties,transpositions=calculate_pos(renners,affinetransformation,args.renners,stitched,fps_scaled,fps,total_transform,indexen,width,baseline,teller)
	#cv2.imwrite("lines32.png",solution)
	#with open('positions32.txt', 'w') as outfile:
    	#	json.dump(renner, outfile)
	cv2.imwrite("solution.jpg",solution)
	return stitched_image,solution,renner,mask,transformaties,renners,transpositions

def process_video(file_name,n=1):
	"""
	iterate over frames of videofile
	:param file_name: filename
	:param n:process one out of every n frames-> default 1: take all frames
	:return:list of finish frames
	"""
	
	fvs= FileVideoStream(file_name).start()					#load video
	
	cam = cv2.VideoCapture(file_name)
	fps = cam.get(cv2.CAP_PROP_FPS)						#get original fps of video
	
	counter=1
	frame_list=[]
	teller=0
	while fvs.running():
		if fvs.more():
			
			teller+=1
			frame= fvs.read()
			
			if frame is not None:
					
					
				frame_list.append(cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2))))   #append frame to list and resize it: height/2, width/2
				
			counter += 1
		else:
			time.sleep(2)
	
	return(frame_list,fps)		
		
def process_frames(frames,numbers,maxgap=12):
	"""
	get the largest group of consecutively finish frames
	:param frames: list of images of possible finish frames
	:param numbers: list of frame indexes of possible finish frames in videostream
	:param maxgap: max gap between indexes of frames to be non-consecutively
	:return: list of finish frames
	"""
	print("process frames")
	groups = [[numbers[0]]]
	indexen=[[0]]
	for i,x in enumerate(numbers[1:]):
    		if abs(x - groups[-1][-1]) <= maxgap:
    			groups[-1].append(x)
    			indexen[-1].append(i+1)
    		else:
    			groups.append([x])
    			indexen.append([i+1])
	index_list=max(indexen, key=len)
	
	return([frames[i] for i in index_list],[numbers[i] for i in index_list[5:]])

	
	


