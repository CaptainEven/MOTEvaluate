"""
2D MOT2016 Evaluation Toolkit
An python reimplementation of toolkit in
2DMOT16(https://motchallenge.net/data/MOT16/)

This file lists the matching algorithms.
1. clear_mot_hungarian: Compute CLEAR_MOT metrics

- Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object
tracking performance: the CLEAR MOT metrics." Journal on Image and Video
 Processing 2008 (2008): 1.

2. idmeasures: Compute MTMC metrics

- Ristani, Ergys, et al. "Performance measures and a data set for multi-target,
 multi-camera tracking." European Conference on Computer Vision. Springer,
  Cham, 2016.



usage:
python evaluate_tracking.py
    --bm                       Whether to evaluate multiple files(benchmarks)
    --seqmap [filename]        List of sequences to be evaluated
    --track  [dirname]         Tracking results directory: default path --
                               [dirname]/[seqname]/res.txt
    --gt     [dirname]         Groundtruth directory:      default path --
                               [dirname]/[seqname]/gt.txt
(C) Yiwen Liu(765305261@qq.com), 2020-10
"""

import sys
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from .bbox import bbox_overlap
from easydict import EasyDict as edict
VERBOSE = False


def clear_mot_hungarian(resDB, gtDB, iou_thresh):
    """
    compute CLEAR_MOT and other metrics
    [recall, precision, FAR, GT, MT, PT, ML, false positives, false negatives,
     id switches, FRA, MOTA, MOTP, MOTAL]
    @res: results
    @gt: fround truth
    """
    # result and gt frame inds(start from 1)
    res_frames = np.unique(resDB[:, 0])
    gt_frames = np.unique(gtDB[:, 0])  

    # result and gt unique IDs
    # either start from 0 or 1 
    res_ids = np.unique(resDB[:, 1])   # result IDs start from 0
    gt_ids = np.unique(gtDB[:, 1])     # gt id start from 1

    # n_frames_gt = int(max(max(res_frames), max(gt_frames)))
    # n_ids_gt = int(max(gt_ids))
    # n_ids_res = int(max(res_ids))
    n_frames_gt = len(gt_frames)
    n_ids_gt = len(gt_ids)
    n_ids_res = len(res_ids)

    # mis-match error for each frame
    mme = np.zeros((n_frames_gt, ), dtype=float)  # ID switch in each frame

    # matches found in each frame
    c = np.zeros((n_frames_gt, ), dtype=float)

    # false positives in each frame
    fp = np.zeros((n_frames_gt, ), dtype=float)

    # missed gts in each frame
    missed = np.zeros((n_frames_gt, ), dtype=float)       

    # gt count in each frame
    gt_count = np.zeros((n_frames_gt, ), dtype=float)

    # overlap matrix(iou matrix)
    d = np.zeros((n_frames_gt, n_ids_gt), dtype=float)   

    # all false positives in all gt frames   
    all_fps = np.zeros((n_frames_gt, n_ids_res), dtype=float)  # account for the number of 1s?

    gt_idx_dicts = [{} for i in range(n_frames_gt)]   # gt frame inds
    res_idx_dicts = [{} for i in range(n_frames_gt)]  # res frame inds

    # matched pairs hashing gt_id to res_id in each frame
    MatchedDicts = [{} for i in range(n_frames_gt)]

    # hash the indices to speed up indexing
    for i in range(gtDB.shape[0]):  # traverse each item(gt bbox)
        frame = np.where(gt_frames == gtDB[i, 0])[0][0]  # original gt track ids(may start from 1)
        gt_id = np.where(gt_ids == gtDB[i, 1])[0][0]  # key: gt_id start from 0
        gt_idx_dicts[frame][gt_id] = i  # i: gt data's item idx

    gt_frames_list = list(gt_frames)
    for i in range(resDB.shape[0]):
        # sometimes detection missed in certain frames, thus should be
        # assigned to ground truth frame id for alignment
        frame = gt_frames_list.index(resDB[i, 0])  # original res track ids(start from 0)
        res_id = np.where(res_ids == resDB[i, 1])[0][0]  # key: res_id start from 0 
        res_idx_dicts[frame][res_id] = i  # i: result data's item idx

    # statistics for each frame(start from the second frame)
    for fr_i in range(n_frames_gt):
        gt_count[fr_i] = len(list(gt_idx_dicts[fr_i].keys()))

        # preserving original mapping if box of this trajectory has large
        #  enough iou in avoid of ID switch
        if fr_i > 0:  # t—(t-1) matching start from the second frame(fr_i = 1)
            mapping_keys = list(MatchedDicts[fr_i - 1].keys())
            mapping_keys.sort()

            for k in range(len(mapping_keys)):
                gt_track_id = mapping_keys[k]  # key: start from 0
                res_track_id = MatchedDicts[fr_i - 1][gt_track_id]  # val: start from 0

                if gt_track_id in list(gt_idx_dicts[fr_i].keys()) and \
                        res_track_id in list(res_idx_dicts[fr_i].keys()):

                    row_gt = gt_idx_dicts[fr_i][gt_track_id]
                    row_res = res_idx_dicts[fr_i][res_track_id]

                    dist = bbox_overlap(resDB[row_res, 2:6], gtDB[row_gt, 2:6])
                    if dist >= iou_thresh:
                        
                        # ----- fill value for Matched matrix
                        MatchedDicts[fr_i][mapping_keys[k]] = res_track_id
                        # -----

                        if VERBOSE:
                            print('perserving mapping: %d to %d' %
                                  (mapping_keys[k], MatchedDicts[fr_i][mapping_keys[k]]))

        # mapping remaining groundtruth and estimated boxes
        unmapped_gt, unmapped_res = [], []
        unmapped_gt = [key for key in gt_idx_dicts[fr_i].keys() if key not in list(MatchedDicts[fr_i].keys())]
        unmapped_res = [key for key in res_idx_dicts[fr_i].keys() if key not in list(MatchedDicts[fr_i].values())]

        if len(unmapped_gt) > 0 and len(unmapped_res) > 0:
            # iou matrix: row: gt, col: res
            overlaps = np.zeros((n_ids_gt, n_ids_res), dtype=float)

            for i in range(len(unmapped_gt)):  # gt 
                row_gt = gt_idx_dicts[fr_i][unmapped_gt[i]]  # row idx(item idx in gt data)

                for j in range(len(unmapped_res)):
                    row_res = res_idx_dicts[fr_i][unmapped_res[j]]  # row idx(item idx in res data)

                    dist = bbox_overlap(resDB[row_res, 2:6], gtDB[row_gt, 2:6])
                    if dist[0] >= iou_thresh:
                        overlaps[i][j] = dist[0]
            
            # hungarian matching: return row_ind(gt), col_ind(res)
            cost_matrix = 1.0 - overlaps
            matched_indices = linear_assignment(cost_matrix)

            for matched in zip(*matched_indices):
                if overlaps[matched[0], matched[1]] == 0:
                    continue
                
                # ----- fill value for Matched matrix,
                #  key: gt track id(start from 0), val: res track id(start from 0)
                MatchedDicts[fr_i][unmapped_gt[matched[0]]] = unmapped_res[matched[1]]
                # -----

                if VERBOSE:
                    print('adding mapping: %d to %d' \
                        % (unmapped_gt[matched[0]], MatchedDicts[fr_i][unmapped_gt[matched[0]]]))

        # compute statistics
        cur_tracked = list(MatchedDicts[fr_i].keys())
        st_tracked = list(MatchedDicts[fr_i].values())
        fps = [key for key in res_idx_dicts[fr_i].keys()
               if key not in list(MatchedDicts[fr_i].values())]

        for k in range(len(fps)):
            all_fps[fr_i][fps[k]] = fps[k]

        # check miss match errors
        if fr_i > 0:
            for i in range(len(cur_tracked)):
                ct = cur_tracked[i]
                est = MatchedDicts[fr_i][ct]
                last_non_empty = -1

                for j in range(fr_i - 1, 0, -1):
                    if ct in MatchedDicts[j].keys():
                        last_non_empty = j
                        break

                if ct in gt_idx_dicts[fr_i - 1].keys() and last_non_empty != -1:
                    mtct, mlastnonemptyct = -1, -1
                    if ct in MatchedDicts[fr_i]:
                        mtct = MatchedDicts[fr_i][ct]
                    if ct in MatchedDicts[last_non_empty]:
                        mlastnonemptyct = MatchedDicts[last_non_empty][ct]

                    if mtct != mlastnonemptyct:
                        mme[fr_i] += 1

        c[fr_i] = len(cur_tracked)
        fp[fr_i] = len(list(res_idx_dicts[fr_i].keys()))
        fp[fr_i] -= c[fr_i]
        missed[fr_i] = gt_count[fr_i] - c[fr_i]

        for i in range(len(cur_tracked)):
            ct = cur_tracked[i]
            est = MatchedDicts[fr_i][ct]
            row_gt = gt_idx_dicts[fr_i][ct]
            row_res = res_idx_dicts[fr_i][est]
            d[fr_i][ct] = bbox_overlap(resDB[row_res, 2:6], gtDB[row_gt, 2:6])

    return mme, c, fp, gt_count, missed, d, MatchedDicts, all_fps


def id_measures(gtDB, stDB, threshold):
    """
    compute MTMC metrics
    [IDP, IDR, IDF1]
    """
    st_ids = np.unique(stDB[:, 1])
    gt_ids = np.unique(gtDB[:, 1])
    n_st = len(st_ids)
    n_gt = len(gt_ids)
    groundtruth = [gtDB[np.where(gtDB[:, 1] == gt_ids[i])[0], :]
                   for i in range(n_gt)]
    prediction = [stDB[np.where(stDB[:, 1] == st_ids[i])[0], :]
                  for i in range(n_st)]
    cost = np.zeros((n_gt + n_st, n_st + n_gt), dtype=float)
    cost[n_gt:, :n_st] = sys.maxsize  # float('inf')
    cost[:n_gt, n_st:] = sys.maxsize  # float('inf')

    fp = np.zeros(cost.shape)
    fn = np.zeros(cost.shape)

    # cost matrix of all trajectory pairs
    cost_block, fp_block, fn_block = cost_between_gt_pred(groundtruth, prediction, threshold)

    cost[:n_gt, :n_st] = cost_block
    fp[:n_gt, :n_st] = fp_block
    fn[:n_gt, :n_st] = fn_block

    # computed trajectory match no groundtruth trajectory, FP
    for i in range(n_st):
        cost[i + n_gt, i] = prediction[i].shape[0]
        fp[i + n_gt, i] = prediction[i].shape[0]

    # groundtruth trajectory match no computed trajectory, FN
    for i in range(n_gt):
        cost[i, i + n_st] = groundtruth[i].shape[0]
        fn[i, i + n_st] = groundtruth[i].shape[0]
    try:
        matched_indices = linear_assignment(cost)
    except:
        import pdb
        pdb.set_trace()
    nbox_gt = sum([groundtruth[i].shape[0] for i in range(n_gt)])
    nbox_st = sum([prediction[i].shape[0] for i in range(n_st)])

    IDFP = 0
    IDFN = 0
    for matched in zip(*matched_indices):
        IDFP += fp[matched[0], matched[1]]
        IDFN += fn[matched[0], matched[1]]
    IDTP = nbox_gt - IDFN
    assert IDTP == nbox_st - IDFP
    IDP = IDTP / (IDTP + IDFP) * 100               # IDP = IDTP / (IDTP + IDFP)
    IDR = IDTP / (IDTP + IDFN) * 100               # IDR = IDTP / (IDTP + IDFN)
    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100

    measures = edict()
    measures.IDP = IDP
    measures.IDR = IDR
    measures.IDF1 = IDF1
    measures.IDTP = IDTP
    measures.IDFP = IDFP
    measures.IDFN = IDFN
    measures.nbox_gt = nbox_gt
    measures.nbox_st = nbox_st

    return measures


def corresponding_frame(traj1, len1, traj2, len2):
    """
    Find the matching position in traj2 regarding to traj1
    Assume both trajectories in ascending frame ID
    """
    p1, p2 = 0, 0
    loc = -1 * np.ones((len1, ), dtype=int)
    while p1 < len1 and p2 < len2:
        if traj1[p1] < traj2[p2]:
            loc[p1] = -1
            p1 += 1
        elif traj1[p1] == traj2[p2]:
            loc[p1] = p2
            p1 += 1
            p2 += 1
        else:
            p2 += 1
    return loc


def compute_distance(traj1, traj2, matched_pos):
    """
    Compute the loss hit in traj2 regarding to traj1
    """
    distance = np.zeros((len(matched_pos), ), dtype=float)
    for i in range(len(matched_pos)):
        if matched_pos[i] == -1:
            continue
        else:
            iou = bbox_overlap(traj1[i, 2:6], traj2[matched_pos[i], 2:6])
            distance[i] = iou
    return distance


def cost_between_trajectories(traj1, traj2, threshold):
    [npoints1, dim1] = traj1.shape
    [npoints2, dim2] = traj2.shape
    # find start and end frame of each trajectories
    start1 = traj1[0, 0]
    end1 = traj1[-1, 0]
    start2 = traj2[0, 0]
    end2 = traj2[-1, 0]

    # check frame overlap
    has_overlap = max(start1, start2) < min(end1, end2)
    if not has_overlap:
        fn = npoints1
        fp = npoints2
        return fp, fn

    # gt trajectory mapping to st, check gt missed
    matched_pos1 = corresponding_frame(
        traj1[:, 0], npoints1, traj2[:, 0], npoints2)
    # st trajectory mapping to gt, check computed one false alarms
    matched_pos2 = corresponding_frame(
        traj2[:, 0], npoints2, traj1[:, 0], npoints1)
    dist1 = compute_distance(traj1, traj2, matched_pos1)
    dist2 = compute_distance(traj2, traj1, matched_pos2)
    # FN
    fn = sum([1 for i in range(npoints1) if dist1[i] < threshold])
    # FP
    fp = sum([1 for i in range(npoints2) if dist2[i] < threshold])
    return fp, fn


def cost_between_gt_pred(groundtruth, prediction, threshold):
    n_gt = len(groundtruth)
    n_st = len(prediction)
    cost = np.zeros((n_gt, n_st), dtype=float)
    fp = np.zeros((n_gt, n_st), dtype=float)
    fn = np.zeros((n_gt, n_st), dtype=float)
    for i in range(n_gt):
        for j in range(n_st):
            fp[i, j], fn[i, j] = cost_between_trajectories(
                groundtruth[i], prediction[j], threshold)
            cost[i, j] = fp[i, j] + fn[i, j]
    return cost, fp, fn


# reference(blog): https://blog.csdn.net/qq_36342854/article/details/102984622
# reference(paper_2008): <<CLEAR Metrics-MOTA&MOTP>>
