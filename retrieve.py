import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import logging
import torch.backends.cudnn as cudnn

from dataset.dataset import DataSet4Test
from tqdm import tqdm
from collections import defaultdict
import torch
from utils.basic_utils import save_json

from standalone_eval.eval import eval_retrieval
from utils.temporal_nms import temporal_non_maximum_suppression

from pipeline.retrievepipe import * 
from pipeline.retrievepipe import get_retrievepipe_class
from pipeline.treebuilder.capTree import CapTree

from utils.model_utils import load_pretrained_models

from config import TestArguments, HfArgumentParser
from dataclasses import asdict

import shutil

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

from utils.basic_utils import seed_it

def filter_vcmr_by_nms(all_video_predictions, nms_threshold=0.6, max_before_nms=1000, max_after_nms=100,
                       score_col_idx=3):
    """ Apply non-maximum suppression for all the predictions for each video.
    1) group predictions by video index
    2) apply nms individually for each video index group
    3) combine and sort the predictions
    Args:
        all_video_predictions: list(sublist),
            Each sublist is [video_idx (int), st (float), ed(float), score (float)]
            Note the scores are negative distances.
        nms_threshold: float
        max_before_nms: int
        max_after_nms: int
        score_col_idx: int
    """
    predictions_neg_by_video_group = defaultdict(list)
    for pred in all_video_predictions[:max_before_nms]:
        predictions_neg_by_video_group[pred[0]].append(pred[1:])  # [st (float), ed(float), score (float)]
    predictions_by_video_group_neg_after_nms = dict()
    for video_idx, grouped_preds in predictions_neg_by_video_group.items():
        predictions_by_video_group_neg_after_nms[video_idx] = temporal_non_maximum_suppression(
            grouped_preds, nms_threshold=nms_threshold)
    predictions_after_nms = []
    for video_idx, grouped_preds in predictions_by_video_group_neg_after_nms.items():
        for pred in grouped_preds:
            pred = [video_idx] + pred  # [video_idx (int), st (float), ed(float), score (float)]
            predictions_after_nms.append(pred)
    # ranking happens across videos, descending order
    predictions_after_nms = sorted(predictions_after_nms, key=lambda x: x[score_col_idx], reverse=True)[:max_after_nms]
    return predictions_after_nms

def post_processing_vcmr_nms(vcmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    vcmr_res: list(dict), each dict is
        {
            "desc": str,
            "desc_id": int,
            "predictions": list(sublist)  # each sublist is
                [video_idx (int), st (float), ed(float), score (float)], video_idx could be different
        }
    """
    processed_vcmr_res = []
    for e in vcmr_res:
        # import pdb; pdb.set_trace() 
        e["predictions"] = filter_vcmr_by_nms(e["predictions"], nms_threshold=nms_thd, max_before_nms=max_before_nms,
                                              max_after_nms=max_after_nms)
        processed_vcmr_res.append(e)
    return processed_vcmr_res

def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap


def eval_q2m(indices, q2m_gts):
    n_q, n_m = indices.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        sorted_idxs = indices[i]
        # sorted_idxs = np.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def eval_q2m_v2(scores, q2m_gts):
    n_q, n_m = scores.shape
    sorted_indices = np.argsort(scores)
    
    gt_list = []
    for i in sorted(q2m_gts):
        gt_list.append(q2m_gts[i][0])
    gt_list = np.array(gt_list)
    pred_ranks = np.argwhere(sorted_indices==gt_list[:, np.newaxis])[:, 1]

    r1 = 100 * (pred_ranks==0).sum() / n_q
    r5 = 100 * (pred_ranks<5).sum() / n_q
    r10 = 100 * (pred_ranks<10).sum() / n_q
    r100 = 100 * (pred_ranks<100).sum() / n_q
    medr = np.median(pred_ranks)
    meanr = pred_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def t2v_map(c2i, t2v_gts):
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in d_i]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


def get_perf(t2v_sorted_indices, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_q2m(t2v_sorted_indices, t2v_gt)
    t2v_map_score = t2v_map(t2v_sorted_indices, t2v_gt)


    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score)


def eval_epoch(pipeline, test_dataset, cfg, epoch=999):
    logger.info("*"*60)
    logger.info('*'*20 + f" Eval epoch: {epoch}" + '*'*20)

    # [Nt, Nv]              
    retrieved_video_indices, vcmr_res_dict = pipeline.retrieval(test_dataset)

    logger.info('video_retrieval_scores:')
    t2v_gt = test_dataset.gt_descid_to_vidid
    video_retrieval_scores = get_perf(retrieved_video_indices, t2v_gt)
    # TMP EXP
    t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = video_retrieval_scores
    vr_json = {
        "r1": round(t2v_r1,1),
        "r5": round(t2v_r5,1),
        "r10": round(t2v_r10,1),
        "r100": round(t2v_r100,1),
        "sumr": round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1),
    }
    # Event-level metrics

    print("evaluaing event-level metrics...")
    IOU_THDS = (0.1, 0.3, 0.5, 0.7)
    

    # # NMS
    vcmr_res_dict_nmsed = dict(video2idx=vcmr_res_dict["video2idx"])
    nms_thd = 0.5 # opt.nms_thd (0.5)
    vcmr_res_dict_nmsed['VCMR'] = post_processing_vcmr_nms(vcmr_res_dict['VCMR'], nms_thd=nms_thd, max_before_nms=1000, max_after_nms=100) # nms_thd=0.5, # max_before_nms=1000

    
    metrics = eval_retrieval(vcmr_res_dict_nmsed, test_dataset.cap_data,
                            iou_thds=IOU_THDS, 
                            use_desc_type=cfg.collection=='tvr') 
    logger.info('-'*40)
    logger.info("VCMR results:")
    vcmr_sum_score = 0
    for k, v in metrics['VCMR'].items():
        vcmr_sum_score += v
        logger.info(str(k) + ': ' + str(v))
    logger.info(f"vcmr_sum_score: {vcmr_sum_score}")
    logger.info('*'*40 + '\n')
    metrics['VCMR']['vcmr_sum_score'] = vcmr_sum_score
    metrics['VR'] = vr_json
    save_metrics_path = os.path.join(cfg.eval_res_dir, f"metrics.json")
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    save_preds_path = os.path.join(cfg.eval_res_dir, "vcmr_preds.json")
    save_json(vcmr_res_dict_nmsed, save_preds_path, save_pretty=True, sort_keys=False)

    return metrics, vcmr_sum_score




def start_inference():
    parser = HfArgumentParser(TestArguments)
    cfg  = parser.parse_args_into_dataclasses()[0]
    print(cfg)
    seed_it(cfg.seed)

    eval_res_dir = os.path.join(cfg.res_dir, cfg.retrieve_dir, cfg.collection, cfg.retrieve_name)
    os.makedirs(eval_res_dir, exist_ok=True)
    if os.path.exists(os.path.join(eval_res_dir, f"metrics.json")):
        print("="*10)
        print(f"metrics.json of {eval_res_dir} already exists, please use another name or delete the existed result")
        return 
    cfg.eval_res_dir = eval_res_dir
    build_settings_path = os.path.join(cfg.res_dir, cfg.construct_dir, cfg.collection, cfg.construct_name, "settings.json")
    assert os.path.exists(build_settings_path), build_settings_path
    shutil.copy(build_settings_path, os.path.join(eval_res_dir, "build_settings.json"))
    build_settings = basic_utils.load_json(build_settings_path)
    eval_settings = asdict(cfg)

    # check
    check_names = ["collection", "num_samples"]
    for name in check_names:
        assert eval_settings[name]==build_settings[name], f"{name} should be same:: build: {build_settings[name]}, eval: {eval_settings[name]}"

    basic_utils.save_json(eval_settings, os.path.join(eval_res_dir, "eval_settings.json"))

    gt_anno_path = os.path.join(cfg.anno_dir, cfg.collection, cfg.anno_file)
    tree_meta_path = os.path.join(cfg.res_dir, cfg.construct_dir, cfg.collection, cfg.construct_name, cfg.tree_file)
    assert os.path.exists(tree_meta_path), tree_meta_path
    cfg.tree_meta_path = tree_meta_path
    print(f"Using tree meta from {cfg.tree_meta_path}")

    pretrained_models = load_pretrained_models(cfg)
    captree = CapTree(cfg, tree_meta_path, pretrained_models)
    test_dataset = DataSet4Test(gt_anno_path, captree_meta=captree.tree_meta)

    captree.compute_tree_feature(resume_video_names=test_dataset.vid_name_to_id.keys())

    infer_pipeline = get_retrievepipe_class(cfg.retrieve_pipeline)(cfg=cfg, captree=captree, models=pretrained_models)
    logger.info("Starting inference...")
    with torch.no_grad():
        eval_epoch(infer_pipeline, test_dataset, cfg)
    
    print(cfg.retrieve_name, "done!")



if __name__ == '__main__':
    start_inference()