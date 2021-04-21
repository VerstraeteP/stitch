import cv2
import numpy as np
from sort import *
import json
from json import JSONEncoder
def calculate_pos(renners,Affinetransform,aantalrenners,afbeelding,fps_scaled,fps,total_transform,indexen,width):
	cnt=0
	aantalrenners=5
	mot_tracker1=Sort(max_age=25, min_hits=1, iou_threshold=0.005)
	track=[]
	rennerspositie={}
	dictrenner=dict()
	renner=[]
	pos_renners=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	Affinetransform=np.asarray(Affinetransform)
	line=[[[0,0],[width,0]]]
	linearray=[]
	dictrenner={"fps":fps,"fps_scaled":fps_scaled}
	transformaties={}

	offset=len(indexen)*width
	for index in range(len(Affinetransform)):
		track.append(mot_tracker1.update(np.array(renners[index])))
		if indexen.count(index-1)==1:
			offset-=width
		"""
		if index>=2:
					
			line =np.array([[[line[0][0][0],line[0][0][1]+300],[line[0][1][0],line[0][1][1]+300]]], dtype = "float32")
						
			
			total=np.vstack((total_transform[index-2],[0,0,1]))
			total[0][2]=0
			total[1][2]=0
			
			
			line=cv2.perspectiveTransform(line,total)
			
			line=cv2.perspectiveTransform(line,np.vstack((Affinetransform[index],[0,0,1])))
			line[0][0][0]+=offset
			
			
		if index==0:
		
			line =np.array([[[line[0][0][0]+500,line[0][0][1]+300],[line[0][1][0]+500,line[0][1][1]+300]]], dtype = "float32")
			line=cv2.perspectiveTransform(line,np.vstack((Affinetransform[index],[0,0,1])))
			print(line)
		if index==1:
		
			line =np.array([[[line[0][0][0],line[0][0][1]+300],[line[0][1][0],line[0][1][1]+300]]], dtype = "float32")
			line=cv2.perspectiveTransform(line,np.vstack((Affinetransform[index],[0,0,1])))
			line[0][0][0]+=offset
			line[0][1][0]+=offset
		
		
		linearray.append(line)
		
		"""

		#transformation=Affinetransform[index]
		for k in track[index]:
			if int(k[4])<=aantalrenners:
				
				
				
				
				if index>=2:
					
					pts =np.array([[[((k[0]+(k[2]-k[0])/2)+300),((k[1]+(k[3]-k[1])/2))+300]]], dtype = "float32")
					
						
					prev=np.vstack((Affinetransform[index-1],[0,0,1]))
					total=np.vstack((total_transform[index-2],[0,0,1]))
					total[0][2]=0
					total[1][2]=0
					prev[0][2]=0
					prev[1][2]=0
					print("k")
					l=cv2.perspectiveTransform(pts,total)
					l=cv2.perspectiveTransform(l,prev)
					l=cv2.perspectiveTransform(l,np.vstack((Affinetransform[index],[0,0,1])))
					l[0][0][0]+=offset
					
				if index==0:
					pts =np.array([[[(k[0]+(k[2]-k[0])/2),(k[1]+(k[3]-k[1])/2)]]], dtype = "float32")

					l=cv2.perspectiveTransform(pts,np.vstack((Affinetransform[index],[0,0,1])))
				if index==1:
					pts =np.array([[[((k[0]+(k[2]-k[0])/2)+300),(k[1]+(k[3]-k[1])/2)+300]]], dtype = "float32")

					l=cv2.perspectiveTransform(pts,np.vstack((Affinetransform[index],[0,0,1])))
					l[0][0][0]+=offset
					
				#right=cv2.perspectiveTransform(bnd_right,transformation)
				#left=cv2.perspectiveTransform(bnd_left,transformation)
				
				
				l= l.astype(int)
				
				#l= l.astype(float)
				#right= right.astype(int)
				#left= left.astype(int)
				#pos_renners[int(k[4])].append(l[0][0].tolist())
				dictrenner1= {"frame_id":index, "position" : l[0][0].tolist()}
				renner.append({"id":k[4],"position":l[0][0].tolist()})
				
				
				if (int(k[4])) in rennerspositie:
					rennerspositie[(int(k[4]))].append(dictrenner1)
				else:
					rennerspositie[(int(k[4]))] = [dictrenner1]
				pos_renners[int(k[4])].append(l[0][0].tolist())
		
		
		dictrenner[index]={"positions":renner}
		renner=[]
	transformaties=Affinetransform.tolist()
	colors=[[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255],[0,0,0],[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255],[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255]]
	
	
	for k in pos_renners:
			
			color=colors[cnt]
			if(len(k)==0):
				pass
			else:
				for index,l in enumerate(k):
					print(l)
					if index == len(k) -1:
						break
					
					cv2.line(afbeelding, (int(l[0]),int(l[1])), (int(k[index + 1][0]),int(k[index + 1][1])), color, 2)
					cv2.circle(afbeelding, (int(l[0]),int(l[1])), 3, (0,0,255), 2)
					
				cnt=cnt+1
	return afbeelding,dictrenner,transformaties
	
