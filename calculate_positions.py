import cv2
import numpy as np
from sort import *
import json
from json import JSONEncoder
def calculate_pos(renners,Affinetransform,aantalrenners,afbeelding,fps_scaled,fps,total_transform,indexen):
	cnt=0
	mot_tracker1=Sort(max_age=25, min_hits=1, iou_threshold=0.005)
	track=[]
	rennerspositie={}
	dictrenner=dict()
	renner=[]
	pos_renners=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	Affinetransform=np.asarray(Affinetransform)
	print("kk")
	print(Affinetransform)
	print(total_transform)
	print(len(renners))
	dictrenner={"fps":fps,"fps_scaled":fps_scaled}
	for k in total_transform:
		k= np.vstack((k,[0,0,1]))
	for k in Affinetransform:
		k= np.vstack((k,[0,0,1]))
	
	offset=300+len(index)*640
	for index in range(len(Affinetransform)):
		track.append(mot_tracker1.update(np.array(renners[index])))
		
		

		#transformation=Affinetransform[index]
		for k in track[index]:
			if int(k[4])<=aantalrenners:
				
				
				bnd_right =np.array([[[k[0],k[1]]]], dtype = "float32")
				bnd_left= np.array([[[k[2],k[3]]]],dtype= "float32")
				
				if index>=2:
					if indexen.count(index)==1:
						offset-=640
					pts =np.array([[[(k[0]+(k[2]-k[0])/2)+offset,(k[1]+(k[3]-k[1])/2)+offset]]], dtype = "float32")
					
						
					prev=np.vstack((Affinetransform[index-1],[0,0,1]))
					total=np.vstack((total_transform[index-2],[0,0,1]))
					total[0][2]=0
					total[1][2]=0
					prev[0][2]=0
					prev[1][2]=0
					print(total)
					print(prev)
					l=cv2.perspectiveTransform(pts,total)
					l=cv2.perspectiveTransform(l,prev)
					l=cv2.perspectiveTransform(l,np.vstack((Affinetransform[index],[0,0,1])))
				if index==0:
					pts =np.array([[[(k[0]+(k[2]-k[0])/2),(k[1]+(k[3]-k[1])/2)]]], dtype = "float32")

					l=cv2.perspectiveTransform(pts,np.vstack((Affinetransform[index],[0,0,1])))
				if index==1:
					pts =np.array([[[(k[0]+(k[2]-k[0])/2+300),(k[1]+(k[3]-k[1])/2)+300]]], dtype = "float32")

					l=cv2.perspectiveTransform(pts,np.vstack((Affinetransform[index],[0,0,1])))
					
				#right=cv2.perspectiveTransform(bnd_right,transformation)
				#left=cv2.perspectiveTransform(bnd_left,transformation)
				
				
				l= l.astype(int)
				print(l)
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
	return afbeelding,dictrenner
	
