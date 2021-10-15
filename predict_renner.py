import json
import os
import random
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg
import glob

# The following is modification of Detectron2 Beginner's Tutorial.
# Cf https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

def get_balloon_dicts(img_dir):
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
        # Pixel-wise segmentation
        

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
                # "Things" are well-defined countable objects,
                # while "stuff" is amorphous something with a different label than the background.
                
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
	
	DatasetCatalog.register("renner_" + d, lambda d=d: get_balloon_dicts("surface_img/" + d))
	MetadataCatalog.get("renner_" + d).set(thing_classes=["renner"])
	balloon_metadata = MetadataCatalog.get("renner_train")
	#dataset_dicts = get_balloon_dicts("surface_img/train")
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("renner_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.0025 #0.00025 # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.OUTPUT_DIR="./drive/MyDrive/renner"


# cfg already contains everything we've set previously. Now we changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)
#dataset_dicts = get_balloon_dicts("balloon/train")
dataset_dicts="surface_img/val"
files=sorted(glob.glob("balloon/val/*.jpg"))
	

def predict_renner(images,masks):
	teller=0
	
	predicted_renners=[]
	counter=0
	for k,img in enumerate(images):
		
		image=img.copy()
		
		image[masks[k] == 255] = 0
		
		
		outputs = predictor(image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
		center=[]
		v= Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)	
		out=v.draw_instance_predictions(outputs['instances'].to("cpu"))
		v=out.get_image()[:, :, ::-1]
		
			
		
		
		for k in range(len(outputs['instances'])):
			counter+=1
			center.append(outputs['instances'].pred_boxes.to("cpu").tensor.numpy()[k].tolist())
			
			center[k].append(outputs['instances'].scores.to("cpu").numpy()[k])
			teller+=1
			
			
			#schrijf renner afbeelding weg"
			position=outputs['instances'].pred_boxes.to("cpu").tensor.numpy()[k].tolist()
			print(position)
			print(v)
			cv2.imwrite("./drive/MyDrive/wkvideo/riders/"+str(k)+str(counter)+".jpg",v[int(position[0]):int(position[2]),int(position[1]):int(position[3])])

			

		predicted_renners.append(center)
		
	
	K.clear_session()
	return predicted_renners

