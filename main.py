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
	
	parser = argparse.ArgumentParser(description='Video-based sprint lane analysis to improve safety in road bicycle racing')
	#parser.add_argument('--video', default="vidbocht.mp4", help='video file to process')
	parser.add_argument('--renners', default=30, help='total renners')
	args = parser.parse_args()
	start_process_video=time.time()
	frame_list,number_list,fps=process_video(video)
	processing_video_time=time.time()-start_process_video
	
	print(len(frame_list))
	scalingfactor=math.ceil(len(frame_list)/300)
	if scalingfactor==0:
		scalingfactor=1
	print(scalingfactor)
	start_stitching_time=time.time()
	stitched_image,affinetransformation,renners,fps_scaled,fps,mask,total_transform,indexen,width=prepare_data_and_stitch(frame_list,fps,scalingfactor)
	stitching_time=time.time()-start_stitching_time
	stitched=stitched_image.copy()
	#cv2.imwrite("solution32.jpg",stitched_image) 
	#print(stitching_time) 
	solution,renner=calculate_pos(renners,affinetransformation,args.renners,stitched,fps_scaled,fps,total_transform,indexen,width)
	#cv2.imwrite("lines32.png",solution)
	#with open('positions32.txt', 'w') as outfile:
    	#	json.dump(renner, outfile)
	cv2.imwrite("solution.jpg",solution)
	return stitched_image,solution,renner,mask 

def process_video(file_name,n=1):
	"""
	iterate over frames of videofile
	:param file_name: filename
	:param n:process one out of every n frames
	:return:list of finish frames
	"""
	print("reading videofile")
	fvs= FileVideoStream(file_name).start()
	
	cam = cv2.VideoCapture(file_name)
	fps = cam.get(cv2.CAP_PROP_FPS)
	
	counter=1
	frame_list=[]
	list_frame_number=[]
	teller=0
	while fvs.running():
		if fvs.more():
			
			teller+=1
			frame= fvs.read()
			
			if frame is not None:
					
					
				frame_list.append(cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2))))
				
			counter += 1
		else:
			time.sleep(2)
	
	return(frame_list,list_frame_number,fps)		
		
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

	
	


