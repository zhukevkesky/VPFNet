import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

 
def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

 

    

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    c0 = 0
    c1 = 0
    c2 = 0
    cnt = 0
    
    iou_dict = []
    score_dict = []
    gt_gt_box = []
    gt_pred_box = []
    image_dict = []
    gt_image_dict = []
    diff_dict = []

    # z_err_dict = []
    # x_err_dict = []
    # y_err_dict = []
 
 
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():

            
            torch.cuda.synchronize()
            start = int(round(time.time()*1000))
            pred_dicts, ret_dict = model(batch_dict)
            torch.cuda.synchronize()
            end = int(round(time.time()*1000))

            if i >= 100:
                c0 += end - start
 
                cnt += 1
                print ('3D backbone time {}ms'.format(c0/cnt))
                # time_dict.append(  np.array(c0)  ) 
       
            # if i >= 100:
            #     c0 += pred_dicts[0]['time1']
            #     c1 += pred_dicts[0]['time2']
            #     c2 +=  pred_dicts[0]['time3']
            #     cnt += 1
            #     print ('3D backbone time {}ms'.format(c0/cnt))
            #     print ('ROI head time {}ms'.format(c1/cnt))
            #     print ("2d backbone time  {}ms".format(c2/cnt))    

    
            # if i >= 100:
            #     # c0 += pred_dicts[0]['seg_time']
            #     c1 += pred_dicts[0]['sp_time']
            #     # c2 +=  pred_dicts[0]['time3']
            #     cnt += 1
            #     # print ('seg time {}ms'.format(c0/cnt))
            #     print ('sp time {}ms'.format(c1/cnt))
            #     # print ("2d backbone time  {}ms".format(c2/cnt))    


        disp_dict = {}
 
        statistics_info(cfg, ret_dict, metric, disp_dict)
        # iou_dict.append( np.array( pred_dicts[0]['gt_bev_IOU']) ) 
        # score_dict.append(   np.array( pred_dicts[0]['gt_pred_score']  )    ) 
        # gt_gt_box.append( np.array( pred_dicts[0]['gt_gt_box']) ) 
        # diff_dict.append (  np.array( pred_dicts[0]['gt_diff'])   ) 
        # gt_pred_box.append( np.array( pred_dicts[0]['gt_pred_box']) ) 
        # image_dict.append(  np.array( pred_dicts[0]['mv_ground_box_2d_left'])  )
        # gt_image_dict.append(  np.array( pred_dicts[0]['gt_box_2d'])  )

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    
    # score_dict = np.concatenate(score_dict ,axis = 0) 
    # gt_pred_box = np.concatenate(gt_pred_box ,axis = 0) 
    # gt_gt_box = np.concatenate(gt_gt_box ,axis = 0) 
    # diff_dict = np.concatenate(diff_dict ,axis = 0) 
    # gt_image_dict = np.concatenate(gt_image_dict ,axis = 0) 
    # iou_dict =  np.concatenate(iou_dict ,axis = 0) 
    # image_dict = np.concatenate(image_dict ,axis = 0) 

    # name = '_rebuttle_voxel'
    # np.save('/root/exchange/origin/gt_pred_score'+ name + '.npy', score_dict) 
    # np.save('/root/exchange/origin/gt_bev_IOU'+ name +'.npy', iou_dict) 
    # np.save('/root/exchange/origin/gt_diff'+ name +'.npy', diff_dict) 
    # np.save('/root/exchange/origin/gt_pred_box'+ name +'.npy', gt_pred_box) 
    # np.save('/root/exchange/origin/image_dict'+ name +'.npy', image_dict) 
    # np.save('/root/exchange/origin/gt_image_dict'+ name +'.npy', gt_image_dict) 
    # np.save('/root/exchange/origin/gt_gt_box'+ name +'.npy', gt_gt_box) 
    # plt.scatter(iou_dict, z_err_dict, alpha=0.6) 
    # plt.savefig('/root/exchange/origin/data.png')


    #         progress_bar.update()
    # print('num of batch', cnt)
    # print ('training time elapsed {}ms'.format(inf_time_sum/cnt))

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
