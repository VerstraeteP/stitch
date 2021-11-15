
from main import start
import os
import glob
import json
import matplotlib.pyplot as plt
import cv2
import sys
from importlib import reload

def main():
	
	for file in glob.glob("./drive/MyDrive/videos/2021/GentWevelgem_1_dames.mp4"): 						#google drive folder with videos
		
			outfiles=[]
			path, wedstrijd = os.path.split(file)									#get all info about race, year,stage, city
			path, jaartal = os.path.split(path)									#from filename
			filename=wedstrijd.split('.')[0]									#....
			wedstrijdnaam=wedstrijd.split('_')[0]									#....
			rit=wedstrijd.split('_')[1]										#....
			aankomstplaats=wedstrijd.split('_')[2].split('.')[0]							#....
			
			stitch,line,renner,mask,transformaties,renners,transposition=start(file)				#call start function from main.py 
			
			json_format = json.dumps(str(renners))
			jsonformat = json.dumps(str(transposition))
			jsonfinal={"Metadata":{"Name":wedstrijdnaam,"Year":jaartal,"Stage":rit,"City":aankomstplaats}} 		#metadata race info
			outfiles.append(jsonfinal)										#combine both metadata + riders positions
			outfiles.append(renner)											#....
			
			
			with open("./drive/MyDrive/dataset/boundingbox/"+str(jaartal)+"/"+filename+".txt", 'w') as outfile: 	#dump boundingboxes of detected riders
				json.dump(json_format, outfile)
			with open("./drive/MyDrive/dataset/transposition/"+str(jaartal)+"/"+filename+".txt", 'w') as outfile:	#dump transformed boundingboxes of detected riders
				json.dump(jsonformat, outfile)
			with open("./drive/MyDrive/dataset/json/"+str(jaartal)+"/"+filename+".txt", 'w') as outfile:		#dump metadata+riders position
				json.dump(outfiles, outfile)
			with open("./drive/MyDrive/dataset/transformaties/"+str(jaartal)+"/"+filename+".txt", 'w') as outfile:	#dump transformation matrices
				json.dump(transformaties, outfile)
			
			cv2.imwrite("./drive/MyDrive/dataset/stitch/"+str(jaartal)+"/"+filename+".jpg",stitch)  		#dump stitched image
			cv2.imwrite("./drive/MyDrive/dataset/lines/"+str(jaartal)+"/"+filename+".jpg",line)			#dump stitched image+ riders lines 
			
					
		
main()
