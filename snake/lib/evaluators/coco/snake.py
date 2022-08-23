import os
import cv2
import json
import numpy as np
from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils
from external.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
import pycocotools.mask as mask_util
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Evaluator:
    ## VPFNet
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']

        self.data_root = args['data_root']
        self.coco = coco.COCO(self.ann_file)
        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
    def depth_to_color(self, depth):
        cmap = plt.cm.jet
        d_min = np.min(depth)
        d_max = np.max(depth)
        depth_relative = (depth - d_min)/(d_max-d_min)
        return 255 * cmap(depth_relative)[:,:,:3]

    def evaluate(self, output, batch):
 
        detection = output['detection']
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio
        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])

        tempfilename =  batch['meta']['tempfilename'][0] 
        path = batch['meta']['path'][0] 
 
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
 
        img = cv2.imread(path) 
        ori_h, ori_w ,_ = img.shape

 
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        
        
        box_points = np.zeros(  (box.shape[0] , 1 , 2 )  ) 
        kkk = 0
        for i in range(box.shape[0]):  
            box_points[kkk, 0  , 0 ] =  (box[i, 2 ] + box[i, 0 ])  / 2  ##### center of bounding box 
            box_points[kkk, 0  , 1 ] =  box[i, 3 ]   ##### bottom of bounding box 
            kkk += 1

 
 
        box_2d = [data_utils.affine_transform(py_, trans_output_inv) for py_ in box_points ] 
        
        box_points = np.zeros(  (box.shape[0] , 1 , 2 + 4 + 3  )  )
        for i in range( len(box_2d) ):
            box_points[i , 0 , :2] = box_2d[i]
            idx_min =  np.argmin( py[i][:,0] , axis=0 )
            idx_max =  np.argmax( py[i][:,0]  , axis=0 )
            idx_min2 =  np.argmin( py[i][:,1]  , axis=0 )
            idx_max2 =  np.argmax( py[i][:,1]  , axis=0 )
       
            box_points[i,0, 2 ]  =   py[i][ idx_min , 0 ]
            box_points[i,0, 4 ]  =   py[i][ idx_max , 0 ]
            box_points[i,0, 3 ]  =   py[i][ idx_min2 , 1 ]
            box_points[i,0, 5 ]  =   py[i][ idx_max2 , 1 ]

        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)
        for i in range( len(rles) ):
                label[i] = self.contiguous_category_id_to_json_id[label[i]]

 
        mask =  snake_eval_utils.poly_to_mask(py, label ,  ori_h, ori_w)
 
        
        coco_dets = []
        segmentation = np.zeros( [ mask[0].shape[0] , mask[0].shape[1]   ] , dtype= np.uint   )
 

        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': int(label[i]) ,
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
 
            coco_dets.append(detection)
        location = np.zeros(len(rles))
      
        ## 1 cyclist
        ## 2 pedestrian
        ## 4 cars
        ## 5 tram
        ## 6 trunck
        ## 7 Van
        ## 8 Misc
    
        for i in range(len(rles)):
            ## filter the bounding box by class, score and box  height
            if  ( (int(label[i]) == 4) or (int(label[i]) == 1) or (int(label[i]) == 2) ) and  (float('{:.2f}'.format(score[i])) > 0.3)   and (  ( np.max(py[i][:,1]) -  np.min(py[i][:,1]) )    > 20  )  : 
                location[i] = np.max(py[i][:,1])   
                box_points[i, 0 , 6] =  int(label[i])  
                box_points[i, 0 , 7] =  (float('{:.2f}'.format(score[i])))
            else:
                location[i] = 1000

 
        foreground_idx = 1 
 
        for i in range(len(rles)):
                    ###  find the closest object by  its 2D location 
                    mask_idx = np.argmin( location ) 
                    if location[mask_idx] < 1000:
                        segmentation = segmentation.reshape(-1)
                        temp_mask = mask[mask_idx].reshape(-1)   
                        temp_mask = temp_mask.astype(bool)
                        location[mask_idx] = 1000
                        segmentation[temp_mask] = foreground_idx
                        box_points[mask_idx, 0 , 8 ] = foreground_idx
                        foreground_idx += 1  
      
        
      
        ###############  for visualization 
        # segmentation = segmentation.astype(np.float)  
        # segmentation[segmentation == 0] = -1.
        # segmentation = segmentation.reshape(  [ mask[0].shape[0] , mask[0].shape[1]   ]  )
        # segmentation = self.depth_to_color(segmentation) 
        # segmentation = Image.fromarray(np.uint8(segmentation))
        # segmentation.save('/data/workspace/zhuhanqi/snake/image_2_box/%s.png' % tempfilename )
        ####################
   
        segmentation = segmentation.reshape(  [ mask[0].shape[0] , mask[0].shape[1] , 1 ]  ) 
        segmentation = segmentation.repeat(3,axis=2)
        segmentation = Image.fromarray(np.uint8(segmentation))
   
        ## save the seg mask and box
        np.save('/data/workspace/zhuhanqi/snake/image_2_box/%s.npy' % tempfilename , box_points )
        segmentation.save('/data/workspace/zhuhanqi/snake/image_2_snake/%s.png' % tempfilename )
 

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)
  

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        
        coco_eval = COCOeval(self.coco, coco_dets, 'segm')
        coco_eval.params.catIds = [4]  
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
         
        return {'ap': coco_eval.stats[0]}

 

Evaluator = Evaluator if cfg.segm_or_bbox == 'segm' else DetectionEvaluator
