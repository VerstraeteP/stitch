import cv2
import numpy as np
from sort import *
import json
from json import JSONEncoder
def calculate_pos(renners,Affinetransform,aantalrenners,afbeelding,fps_scaled,fps,total_transform,indexen,width,baseline):
	cnt=0
	mot_tracker1=Sort(max_age=500000, min_hits=3, iou_threshold=0.005)#thresh 0.005
	mot_tracker1.reset_count()

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
	
	offset=0
	for k in baseline:
		offset+=k
	for index in range(len(Affinetransform)):
		track.append(mot_tracker1.update(np.array(renners[index])))
		if indexen.count(index-1)==1:
			k=baseline.pop(len(baseline)-1)
			offset-=k
		"""
		if index>=2:
					
			line =np.array([[[line[0][0][0]+300,line[0][0][1]+300],[line[0][1][0]+300,line[0][1][1]+300]]], dtype = "float32")
						
			prev=np.vstack((Affinetransform[index-1],[0,0,1]))
			total=np.vstack((total_transform[index-2],[0,0,1]))
			total[0][2]=0
			total[1][2]=0
			prev[0][2]=0
			prev[1][2]=0
			
			copy=Affinetransform[index].copy()
			l=cv2.perspectiveTransform(line,total)
			l=cv2.perspectiveTransform(l,prev)
			l=cv2.perspectiveTransform(l,np.vstack((copy,[0,0,1])))
			l[0][0][0]+=offset
			l[0][1][0]+=offset
			
			
		if index==0:
		
			line =np.array([[[line[0][0][0]+500,line[0][0][1]],[line[0][1][0]+500,line[0][1][1]]]], dtype = "float32")
			l=cv2.perspectiveTransform(line,np.vstack((Affinetransform[index],[0,0,1])))
			
		if index==1:
		
			line =np.array([[[line[0][0][0]+300,line[0][0][1]+300],[line[0][1][0]+300,line[0][1][1]+300]]], dtype = "float32")
			copy=Affinetransform[index].copy()

			l=cv2.perspectiveTransform(line,np.vstack((copy,[0,0,1])))
			l[0][0][0]+=offset
			l[0][1][0]+=offset
		
		
		linearray.append(l)
		"""
		

		#transformation=Affinetransform[index]
		for k in track[index]:
			if int(k[4])<=aantalrenners:
				
				
				
				
				if index>=2:
					bounding1=np.array([[[k[0],k[1]]]], dtype = "float32")
					bounding2=np.array([[[k[2],k[3]]]], dtype = "float32")
					pts =np.array([[[((k[0]+(k[2]-k[0])/2)+300),((k[1]+(k[3]-k[1])/2))+300]]], dtype = "float32")
					
						
					prev=np.vstack((Affinetransform[index-1],[0,0,1]))
					total=np.vstack((total_transform[index-2],[0,0,1]))
					total[0][2]=0
					total[1][2]=0
					prev[0][2]=0
					prev[1][2]=0
					
					copy=Affinetransform[index].copy()
					l=cv2.perspectiveTransform(pts,total)
					bounding1=cv2.perspectiveTransform(bouding1,total)
					bounding2=cv2.perspectiveTransform(bouding2,total)

					l=cv2.perspectiveTransform(l,prev)
					bounding1=cv2.perspectiveTransform(bounding1,prev)
					bounding2=cv2.perspectiveTransform(bounding2,prev)
					l=cv2.perspectiveTransform(l,np.vstack((copy,[0,0,1])))
					bounding1=cv2.perspectiveTransform(bounding1,np.vstack((copy,[0,0,1])))
					bounding2=cv2.perspectiveTransform(bounding2,np.vstack((copy,[0,0,1])))
					l[0][0][0]+=offset
					bounding1[0][0][0]+=offset
					bounding2[0][0][0]+=offset
					
				
				if index==1:
					bounding1=np.array([[[k[0],k[1]]]], dtype = "float32")
					bounding2=np.array([[[k[2],k[3]]]], dtype = "float32")
					pts =np.array([[[((k[0]+(k[2]-k[0])/2)+300),(k[1]+(k[3]-k[1])/2)+300]]], dtype = "float32")
					copy=Affinetransform[index].copy()

					l=cv2.perspectiveTransform(pts,np.vstack((copy,[0,0,1])))
					bounding1=cv2.perspectiveTransform(bounding1,np.vstack((copy,[0,0,1])))
					bounding2=cv2.perspectiveTransform(bounding2,np.vstack((copy,[0,0,1])))
					l[0][0][0]+=offset
					bounding1[0][0][0]+=offset
					bounding2[0][0][0]+=offset
					
				#right=cv2.perspectiveTransform(bnd_right,transformation)
				#left=cv2.perspectiveTransform(bnd_left,transformation)
				if index==0:
					pass
				else:
				
					l= l.astype(int)

					#l= l.astype(float)
					#right= right.astype(int)
					#left= left.astype(int)
					#pos_renners[int(k[4])].append(l[0][0].tolist())
					dictrenner1= {"frame_id":index, "position" : l[0][0].tolist()}
					renner.append({"id":k[4],"position":l[0][0].tolist(),"boundingbox":[bounding1[0][0].tolist(),bounding2[0][0].tolist()]})


					if (int(k[4])) in rennerspositie:
						rennerspositie[(int(k[4]))].append(dictrenner1)
					else:
						rennerspositie[(int(k[4]))] = [dictrenner1]
					pos_renners[int(k[4])].append(l[0][0].tolist())
		
		
		dictrenner[index]={"positions":renner}
		renner=[]
	transformaties=Affinetransform.tolist()
	
	colors=[[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255],[0,0,0],[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255],[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255]]
	"""
	for k in linearray:
		cv2.line(afbeelding, (int(k[0][0][0]),int(k[0][0][1])), (int(k[0][1][0]),int(k[0][1][1])), colors[5], 4)
	"""
	
	for k in pos_renners:
			
			color=colors[cnt]
			if(len(k)==0):
				pass
			else:
				for index,l in enumerate(k):
					
					if index == len(k) -1:
						break
					
					cv2.line(afbeelding, (int(l[0]),int(l[1])), (int(k[index + 1][0]),int(k[index + 1][1])), color, 2)
					cv2.circle(afbeelding, (int(l[0]),int(l[1])), 3, (0,0,255), 2)
					
				cnt=cnt+1
	return afbeelding,dictrenner,transformaties
	
