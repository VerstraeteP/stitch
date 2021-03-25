import json
import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg
from backgroundsubtraction import backgroundsubtraction
import glob
import math



def get_surface_dicts(img_dir):
	"""
	convert dataset to coco-format
	:param img_dir: direction of images
	:return: dict of annotations 
	"""
	json_file = os.path.join(img_dir, "via_region_data.json")
	with open(json_file) as f:
		imgs_anns = json.load(f)
	dataset_dicts = []
	for idx, v in enumerate(imgs_anns.values()):
		record = {}
		filename = os.path.join(img_dir, v["filename"])
		height, width = cv2.imread(filename).shape[:2]
		record["file_name"] = filename
		record["image_id"] = idx
		record["height"] = height
		record["width"] = width
		annos = v["regions"]
		objs = []
		for annotation in annos:
			anno = annotation["shape_attributes"]
			px = anno["all_points_x"]
			py = anno["all_points_y"]
			poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
			poly = [p for x in poly for p in x]
			obj = {
				"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
				"bbox_mode": BoxMode.XYXY_ABS,
				"segmentation": [poly],
				"category_id": 0,
				}
			objs.append(obj)
			record["annotations"] = objs
			dataset_dicts.append(record)
		
	return dataset_dicts
	
def predict_surface(img):
	
	for d in ["train", "val"]:
		
			DatasetCatalog.register("surface_" + d, lambda d=d: get_surface_dicts("surface/" + d))
			MetadataCatalog.get("surface_" + d).set(thing_classes=["surface"])
			surface_metadata = MetadataCatalog.get("surface_train")
			
		
	cfg = get_cfg()
	
		
		
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	cfg.DATASETS.TRAIN = ("surface_train",)
	cfg.DATASETS.TEST = ()
	cfg.DATALOADER.NUM_WORKERS = 2
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 
	cfg.SOLVER.IMS_PER_BATCH = 1
	cfg.SOLVER.BASE_LR = 0.0025 
	cfg.SOLVER.MAX_ITER = 1000    
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
	cfg.OUTPUT_DIR="./drive/MyDrive/surface"
	"""
		os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
		trainer = DefaultTrainer(cfg) 
		trainer.resume_or_load(resume=False)
		trainer.train()
	"""
	
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set a custom testing threshold
	predictor = DefaultPredictor(cfg)
	#dataset_dicts = get_balloon_dicts("balloon/train")
	dataset_dicts="surface_img/val"
	files=sorted(glob.glob("balloon/val/*.jpg"))
	
	teller=1
	data=[]
		
	middle=[img[0].shape[0]/2,img[0].shape[1]/2]
	background=backgroundsubtraction(img)
	cv2.imwrite("background.jpg",background)
	
	#print(background.shape)
	
		
		
	for k in img:
		
		minimum=None
		predictor = DefaultPredictor(cfg)
		outputs = predictor(k)
		
		v = Visualizer(k[:, :, ::-1], metadata=surface_metadata, scale=0.5)
		out=v.draw_instance_predictions(outputs["instances"].to("cpu"))
		v=out.get_image()[:, :, ::-1]
		plt.imshow(v),plt.title("Warped Image")
		
		
		
		#maskoutput=outputs['instances'].pred_masks.to("cpu")[0][:2]
		maskoutput=0
		
		

		
		for k in outputs['instances'].pred_masks.to("cpu").numpy():
			maskoutput+=k
	
		maskoutput=maskoutput*255
		maskoutput+=background
		
		maskoutput = maskoutput.astype(np.uint8)
		mask = np.ones((k.shape[0], k.shape[1]), dtype=np.uint8) 
		img_res = cv2.bitwise_and(mask,mask, mask = maskoutput)
  
		data.append(img_res)
	return data
    
    
   
if __name__ == "__main__":
	main()
