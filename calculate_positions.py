import cv2
import numpy as np
from sort import *
import json
from json import JSONEncoder
def calculate_pos(renners,Affinetransform,aantalrenners,afbeelding,fps_scaled,fps,total_transform,indexen,width,baseline,counter):
	"""
	calculate the positions of the riders, using the detected positions
	
	
	"""
	teller=0
	aantalrenners=10000							#parameter to only write the first x positions (here large number, so all positions)
	
	teller+=1
	mot_tracker1=Sort(max_age=500000, min_hits=3, iou_threshold=0.005)  	#initialize sort-tracker algorithm
	#mot_tracker1.reset_count()

	track=[]
	rennerspositie={}
	dictrenner=dict()
	renner=[]
	pos_renners=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	trans_position=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	Affinetransform=np.asarray(Affinetransform)
	line=[[[0,0],[width,0]]]
	linearray=[]
	dictrenner={"fps":fps,"fps_scaled":fps_scaled}
	transformaties={}
	teller=-1
	offset=0
	for k in baseline:
		offset+=k
	
	for index in range(len(Affinetransform[:counter])):
		
		if len(np.array(renners[index]))!=0:						#if detected renners in frame are not 0
			track.append(mot_tracker1.update(np.array(renners[index])))		#add to mot_tracker, renners[index] contains all boundingboxes + prediction score per frame
			teller+=1
			if indexen.count(index-1)==1:						#if the index is in the list of indexes, so we did a translation, we take the offset into account
				k=baseline.pop(len(baseline)-1)
				offset-=k
			
			transformation=Affinetransform[index]					#get transformation
		if len(track[teller])==0:							#if no rider is detected in the frame add "null" to dict

			renner.append({"id":"null","position":"null","boundingbox":"null"})
		else:										
			for k in track[teller]:
				if int(k[4])<=aantalrenners:




							if index>=2:
								bounding1=np.array([[[k[0]+300,k[1]+300]]], dtype = "float32")		#calculate new boundingboxcoordinates,we add 300 to both x and y because in the stitching we also made the same shift
								bounding2=np.array([[[k[2]+300,k[3]+300]]], dtype = "float32")
								pts =np.array([[[((k[0]+(k[2]-k[0])/2)+300),((k[1]+(k[3]-k[1])/2))+300]]], dtype = "float32") # the riders position: middle of the bounding box+ offset of 300 
								rennerx=np.array([[[k[0]+300,k[1]+300]]], dtype= "float32")		#get x and y coordinates of the rider
								rennery=np.array([[[k[2]+300,k[3]+300]]],dtype="float32")


								prev=np.vstack((Affinetransform[index-1],[0,0,1]))			#get last transformation
								total=np.vstack((total_transform[index-2],[0,0,1]))			#get total transformation until the previous image
								total[0][2]=0
								total[1][2]=0
								prev[0][2]=0
								prev[1][2]=0

								copy=Affinetransform[index].copy()
								l=cv2.perspectiveTransform(pts,total)					
								rennerlx=cv2.perspectiveTransform(rennerx,total)			#calculate the new position using the total transformation
								rennerly=cv2.perspectiveTransform(rennery,total)			#same but in y direction
								bounding1=cv2.perspectiveTransform(bounding1,total)			#calculate the new bounding position
								bounding2=cv2.perspectiveTransform(bounding2,total)

								l=cv2.perspectiveTransform(l,prev)					
								rennerlx=cv2.perspectiveTransform(rennerlx,prev)		
								rennerly=cv2.perspectiveTransform(rennerly,prev)
								bounding1=cv2.perspectiveTransform(bounding1,prev)
								bounding2=cv2.perspectiveTransform(bounding2,prev)
								l=cv2.perspectiveTransform(l,np.vstack((copy,[0,0,1])))
							
								rennerlx=cv2.perspectiveTransform(rennerlx,np.vstack((copy,[0,0,1])))
								rennerly=cv2.perspectiveTransform(rennerly,np.vstack((copy,[0,0,1])))		 

								bounding1=cv2.perspectiveTransform(bounding1,np.vstack((copy,[0,0,1])))
								bounding2=cv2.perspectiveTransform(bounding2,np.vstack((copy,[0,0,1])))
								l[0][0][0]+=offset							#add the offset of the shift we made if we expand our image at the left side
								rennerlx[0][0][0]+=offset
								rennerly[0][0][0]+=offset
								bounding1[0][0][0]+=offset
								bounding2[0][0][0]+=offset


							if index==1:								# if index==1: so second image: same as above but no total transformation  
								bounding1=np.array([[[k[0]+300,k[1]+300]]], dtype = "float32")
								bounding2=np.array([[[k[2]+300,k[3]+300]]], dtype = "float32")
								pts =np.array([[[((k[0]+(k[2]-k[0])/2)+300),(k[1]+(k[3]-k[1])/2)+300]]], dtype = "float32")
								rennerx=np.array([[[k[0]+300,k[1]+300]]], dtype= "float32")
								rennery=np.array([[[k[2]+300,k[3]+300]]],dtype="float32")

								copy=Affinetransform[index].copy()

								l=cv2.perspectiveTransform(pts,np.vstack((copy,[0,0,1])))
								rennerlx=cv2.perspectiveTransform(rennerx,np.vstack((copy,[0,0,1])))
								rennerly=cv2.perspectiveTransform(rennery,np.vstack((copy,[0,0,1])))
								bounding1=cv2.perspectiveTransform(bounding1,np.vstack((copy,[0,0,1])))
								bounding2=cv2.perspectiveTransform(bounding2,np.vstack((copy,[0,0,1])))
								l[0][0][0]+=offset
								rennerlx[0][0][0]+=offset
								rennerly[0][0][0]+=offset
								bounding1[0][0][0]+=offset
								bounding2[0][0][0]+=offset

							#right=cv2.perspectiveTransform(bnd_right,transformation)
							#left=cv2.perspectiveTransform(bnd_left,transformation)
							if index==0:
								pass
							else:										#write all the positions to a dict

								l= l.astype(int)
								rennerlx=rennerlx.astype(int)
								rennerly=rennerly.astype(int)

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
								rennerend=[]	
								
								rennerend.append(rennerlx[0][0].tolist())
								rennerend.append(rennerly[0][0].tolist())
								pos_renners[int(k[4])].append(l[0][0].tolist())
								trans_position[int(k[4])].append(rennerend)

				
		
		dictrenner[index]={"positions":renner}
		renner=[]
	transformaties=Affinetransform.tolist()
	
	colors=[[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255],[0,0,0],[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255],[0,0,0],[0,255,255],[255,255,2],[255,1,2],[132,125,25],[255,125,0],[255,0,255],[0,0,0],[0,255,0],[255,125,2],[255,1,2],[125,125,125],[255,125,0],[255,0,255]]
	"""
	for k in linearray:
		cv2.line(afbeelding, (int(k[0][0][0]),int(k[0][0][1])), (int(k[0][1][0]),int(k[0][1][1])), colors[5], 4)
	"""
	cnt=0
	for k in pos_renners:					#draw positions of riders
			
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
	return afbeelding,dictrenner,transformaties,trans_position
	
