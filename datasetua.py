
from main import start

import os
import glob
import json
import matplotlib.pyplot as plt
import cv2
import sys
from importlib import reload

def main():
	
	for file in glob.glob("/drive/MyDrive/videos/*/*"):
		
		try:
			outfiles=[]
			print(file)
			
			path, wedstrijd = os.path.split(file)
			path, jaartal = os.path.split(path)
			filename=wedstrijd.split('.')[0]
			wedstrijdnaam=wedstrijd.split('_')[0]
			img=cv2.imread("jersey.png")
			rit=wedstrijd.split('_')[1]
			aankomstplaats=wedstrijd.split('_')[2].split('.')[0]
			
			jsonfinal={"Metadata":{"Name":wedstrijdnaam,"Year":jaartal,"Stage":rit,"City":aankomstplaats}}
			stitch,line,renner,mask=start(file)
			
			
			cv2.imwrite("/drive/MyDrive/dataset/stitch/"+str(jaartal)+"/"+filename+".jpg",stitch)
			cv2.imwrite("/drive/MyDrive/dataset/lines/"+str(jaartal)+"/"+filename+".jpg",line)
			cv2.imwrite("/drive/MyDrive/dataset/mask/"+str(jaartal)+"/"+filename+".jpg",mask)
			outfiles.append(jsonfinal)
			outfiles.append(renner)
			with open("/drive/MyDrive/dataset/json/"+str(jaartal)+"/"+filename+".txt", 'w') as outfile:
				json.dump(outfiles, outfile)
			
		except:
			print("failed")
			continue
			
		
main()
