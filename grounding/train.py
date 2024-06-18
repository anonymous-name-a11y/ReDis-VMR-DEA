import argparse
import numpy
import logging
import sys
import os
import time
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.model_saver import ModelSaver
from util.helper_function import set_device, StatisticsPrint, LoggerInfo, update_values, group_weight
from model.SpanGroundMatchDisc import GMD
from loss import temporal_order_discrimination_loss, span_ground_loss, BCE_loss, \
    matching_KL_divergence, span_pred, compute_mean_iou
from model.networks.attention import masked_softmax

from dataset.bias import BiasDataGenerator

def perpare_data(batch_data):
    sent_list, sent_feat, sent_len, sent_mask, \
    video_duration, vid_list, \
    ori_video_feat, ori_nfeats, ori_video_mask, ori_gt, \
    pseudo_video_feat, pseudo_nfeats, pseudo_video_mask, pseudo_gt = batch_data

    sent_feat = sent_feat.cuda()
    sent_mask = sent_mask.cuda()
    ori_video_feat = ori_video_feat.cuda()
    ori_video_mask = ori_video_mask.cuda()

    pseudo_video_feat = pseudo_video_feat.cuda()
    pseudo_video_mask = pseudo_video_mask.cuda()

    # used for temporal order discrimination
    keys = ['temporal_labels', 'fore_masks', 'back_masks']
    for k in keys:
        ori_gt[k] = ori_gt[k].cuda()
        pseudo_gt[k] = pseudo_gt[k].cuda()

    return sent_list, sent_feat, sent_len, sent_mask, \
            video_duration, vid_list, \
            ori_video_feat, ori_nfeats, ori_video_mask, ori_gt, \
            pseudo_video_feat, pseudo_nfeats, pseudo_video_mask, pseudo_gt

def constract_model(params, logger):
    # if params['start_from'] is not None:
    #     state_dict = torch.load(params['start_from'],
    #                             map_location=lambda storage, location: storage)
    #     logger.warn('use checkpoint: %s', params['start_from'])

    video_seq_set = {}
    video_seq_set['name'] = params['video_encoder']
    video_seq_set['input_dim'] = params['video_feature_dim']
    video_seq_set['rnn_hidden_dim'] = params['video_rnn_hiddendim']
    video_seq_set['rnn_layers'] = params['video_rnn_layers']
    video_seq_set['rnn_cell'] = params['video_rnn_cell']
    video_seq_set['mask'] = params['mask']
    video_seq_set['drop_out'] = params['dropout']
    video_seq_set['T'] = params['video_len']
    video_seq_set['nblocks'] = 2

    sent_seq_set = {}
    sent_seq_set['name'] = params['sent_encoder']
    sent_seq_set['input_dim'] = 300
    sent_seq_set['rnn_hidden_dim'] = params['sent_rnn_hiddendim']
    sent_seq_set['rnn_layers'] = params['sent_rnn_layers']
    sent_seq_set['rnn_cell'] = params['sent_rnn_cell']
    sent_seq_set['drop_out'] = params['dropout']

    # grounding
    grounding_set = {}
    grounding_set['cross_name'] = params['crossmodal']
    grounding_set['name'] = params['predictor']
    grounding_set['lstm_hidden_dim'] = params['span_hidden_dim']
    grounding_set['mlp_hidden_dim'] = params['mlp_hidden_dim']

    # matching
    matching_set= {}
    cross_set, temporal_set, predict_set = {}, {}, {}
    cross_set['name'] = params['m_cross']
    temporal_set['name'] = params['m_temp']
    temporal_set['hidden_dim'] = 256
    temporal_set['layers'] = 2
    temporal_set['dropout'] = params['dropout']

    predict_set['name'] = params['m_pred']
    predict_set['activation'] = params['m_pred_activ']
    predict_set['hidden_dim'] = params['m_pred_hidden']

    matching_set['cross'] = cross_set
    matching_set['temporal'] = temporal_set
    matching_set['predict'] = predict_set

    model = GMD(video_seq_set, sent_seq_set, grounding_set, matching_set, logger, params['dropout'])

    logger.info('*' * 120)
    sys.stdout.flush()
    print('Model' + '*' * 110)
    print(model)

    # if params['start_from'] is not None:
    #     model.load_state_dict(state_dict)
    #     print("load over.", params['start_from'])

    return model

def train(model, data_loader, params, logger, step, optimizer, criterion_domain, dataset, bias_g, bias_ps, bias_see):
    model.train()

    _start_time = time.time()
    accumulate_loss = 0
    accumulate_iou = 0
    accumulate_loss_g = 0
    accumulate_loss_m1 = 0
    accumulate_loss_m2 = 0
    accumulate_loss_d = 0

    logger.info('learning rate:' + '*' * 106)
    logger.info('training on optimizing localizing')
    for param_group in optimizer.param_groups:
        logger.info('  ' * 7 + '|: lr %s, wd %s', param_group['lr'], param_group['weight_decay'])
    logger.info('*' * 120)

    # exit(0)

    stamp_result = []

    for idx, batch_data in enumerate(data_loader):
        batch_time = time.time()


        # print('video_duration', video_duration)
        # print('vid_list', vid_list)
        # print('ori_video_feat', ori_video_feat.shape)
        # print('ori_video_feat', ori_video_feat[0])
        # print('ori_nfeats', ori_nfeats)
        # print('ori_video_mask', ori_video_mask[0])
        # for k, v in ori_gt.items():
        #     print(k, v[0])
        # exit(0)

        sent_list, sent_feat, sent_len, sent_mask, \
        video_duration, vid_list, \
        ori_video_feat, ori_nfeats, ori_video_mask, ori_gt, \
        pseudo_video_feat, pseudo_nfeats, pseudo_video_mask, pseudo_gt = perpare_data(batch_data)

        # if step < 15 - 1:
        # if step < 0:
        if True:
            ori_span_prob, ori_match_prob, pseudo_match_prob, \
                ori_disc_prob, pseudo_disc_prob = model(
                sent_feat, sent_mask,
                ori_video_feat, ori_video_mask,
                pseudo_video_feat, pseudo_video_mask,
                ori_gt['temporal_labels'], ori_gt['fore_masks'], ori_gt['back_masks'],
                pseudo_gt['temporal_labels'], pseudo_gt['fore_masks'], pseudo_gt['back_masks'],
            )

            # LOSS
            # Grounding Loss
            loss_g = span_ground_loss(ori_span_prob['start'], ori_span_prob['end'], ori_gt['framestps'])

            # Cross-Modal Semantic Matching Loss
            #   m1: intra-video
            loss_intra = params['loss_m1_lambda'] * ( \
                        BCE_loss(ori_match_prob, ori_gt['temporal_labels'], ori_video_mask) + \
                        BCE_loss(pseudo_match_prob, pseudo_gt['temporal_labels'], pseudo_video_mask))

            #   m2: inter-videos
            ori_mask = ori_gt['temporal_labels']
            pseudo_mask = pseudo_gt['temporal_labels']
            ori_match_prob = masked_softmax(ori_match_prob, ori_mask.cuda())
            pseudo_match_prob = masked_softmax(pseudo_match_prob, pseudo_mask.cuda())

            # ori和pseudo时间片段长度不一样的话，就不能计算这个loss
            loss_inter = params['loss_m2_lambda'] * matching_KL_divergence(
                ori_match_prob, pseudo_match_prob,
                ori_gt['framestps'], pseudo_gt['framestps']
            )
            # loss_inter = torch.Tensor([0.]).to(ori_match_prob.device)

            # Temporal Order Discrimination Loss
            loss_disc = temporal_order_discrimination_loss(ori_disc_prob, pseudo_disc_prob,
                                                           criterion_domain)


            loss_ori = loss_g + loss_intra + loss_inter + params['loss_disc_lambda'] * loss_disc

            # loss = loss_g + loss_intra + loss_inter + params['loss_disc_lambda'] * loss_disc
            # optimizer.zero_grad()
            # loss.backward()
            # if params['grad_clip']:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['grad_clip_max'], norm_type=2)
            # optimizer.step()

        # else:
        if True:

            (new_feats, new_mask, new_mask_label, new_mask_fore, new_mask_back, new_sampled_points,
             aug_framestamps, aug_nfeats, aug_video_feature, new_mask_label_p, new_mask_fore_p, new_mask_back_p) = bias_g.sample_point(
                vid_list, ori_video_feat, ori_nfeats, ori_video_mask, ori_gt['framestps']
            )


            # print(ori_gt['framestps'], new_sampled_points)
            # for iii, xxx in enumerate(new_feats[0]):
            #     if ori_gt['framestps'][iii][1] > ori_nfeats[iii]:
            #         print(vid_list[iii], ori_nfeats[iii], ori_gt['framestps'][iii][1])
            #         raise ValueError((ori_nfeats[iii], ori_gt['framestps'][iii][1]))
            #
            #     print(iii, xxx[0], ori_video_feat[0][iii][0], new_mask[0][iii])
            # exit()

            ori_video_feat = new_feats
            ori_video_mask = new_mask
            ori_gt['temporal_labels'] = new_mask_label
            ori_gt['fore_masks'] = new_mask_fore
            ori_gt['back_masks'] = new_mask_back
            ori_gt['framestps'] = new_sampled_points

            stamp_result.append(new_sampled_points)

            # new_feats, new_mask, new_mask_label, new_mask_fore, new_mask_back, new_sampled_points, _, _, _, _, _, _ = bias_ps.sample_point(
            #     vid_list, pseudo_video_feat, pseudo_nfeats, pseudo_video_mask
            # )
            # pseudo_video_feat = aug_video_feature
            # pseudo_video_mask = new_mask
            # pseudo_gt['temporal_labels'] = new_mask_label_p
            # pseudo_gt['fore_masks'] = new_mask_fore_p
            # pseudo_gt['back_masks'] = new_mask_back_p
            # pseudo_gt['framestps'] = aug_framestamps

            # pseudo_video_feat = new_feats
            # pseudo_video_mask = new_mask
            # pseudo_gt['temporal_labels'] = new_mask_label
            # pseudo_gt['fore_masks'] = new_mask_fore
            # pseudo_gt['back_masks'] = new_mask_back
            # pseudo_gt['framestps'] = new_sampled_points

            ori_span_prob_old = ori_span_prob

            ori_span_prob, ori_match_prob, pseudo_match_prob, \
            ori_disc_prob, pseudo_disc_prob = model(
                sent_feat, sent_mask,
                ori_video_feat, ori_video_mask,
                pseudo_video_feat, pseudo_video_mask,
                ori_gt['temporal_labels'], ori_gt['fore_masks'], ori_gt['back_masks'],
                pseudo_gt['temporal_labels'], pseudo_gt['fore_masks'], pseudo_gt['back_masks'],
            )

            # ori_span_prob['start'] = ori_span_prob_old['start'] + 0.01 * ori_span_prob['start']
            # ori_span_prob['end'] = ori_span_prob_old['end'] + 0.01 * ori_span_prob['end']

            # LOSS
            # Grounding Loss
            loss_g = span_ground_loss(ori_span_prob['start'], ori_span_prob['end'], ori_gt['framestps'])

            # Cross-Modal Semantic Matching Loss
            #   m1: intra-video
            loss_intra = params['loss_m1_lambda'] * ( \
            BCE_loss(ori_match_prob, ori_gt['temporal_labels'], ori_video_mask) + \
            BCE_loss(pseudo_match_prob, pseudo_gt['temporal_labels'], pseudo_video_mask) )


            #   m2: inter-videos
            ori_mask = ori_gt['temporal_labels']
            pseudo_mask = pseudo_gt['temporal_labels']
            ori_match_prob = masked_softmax(ori_match_prob, ori_mask.cuda())
            pseudo_match_prob = masked_softmax(pseudo_match_prob, pseudo_mask.cuda())

            # ori和pseudo时间片段长度不一样的话，就不能计算这个loss
            # loss_inter = params['loss_m2_lambda'] * matching_KL_divergence(
            #     ori_match_prob, pseudo_match_prob,
            #     ori_gt['framestps'], pseudo_gt['framestps']
            # )
            loss_inter = torch.Tensor([0.]).to(ori_match_prob.device)

            # Temporal Order Discrimination Loss
            loss_disc = temporal_order_discrimination_loss(ori_disc_prob, pseudo_disc_prob,
                                                        criterion_domain)
            loss = loss_g + loss_intra # + loss_inter + params['loss_disc_lambda'] * loss_disc



        loss = 0.01 * loss + loss_ori  # best
        # loss = 0.1 * loss + loss_ori  #
        # loss = 1 * loss + loss_ori  #


        optimizer.zero_grad()
        loss.backward()
        if params['grad_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['grad_clip_max'], norm_type=2)
        optimizer.step()

        # statics
        pred_time, score = span_pred(ori_span_prob['start'].cpu(), ori_span_prob['end'].cpu())
        pred_time = dataset.frame2sec(pred_time.float(), duration= video_duration, nfeats= ori_nfeats)
        miou = compute_mean_iou(pred_time.float().data, ori_gt['timestps'].data)

        accumulate_iou += miou.cpu().item()
        accumulate_loss += loss.cpu().item()
        accumulate_loss_g += loss_g.cpu().item()
        accumulate_loss_m1 += loss_intra.cpu().item()
        accumulate_loss_m2 += loss_inter.cpu().item()
        accumulate_loss_d += loss_disc.cpu().item()

        if params['batch_log_interval'] != -1 and idx % params['batch_log_interval'] == 0:
            logger.info(
                'train: epoch[%03d], batch[%04d/%04d], elapsed time=%0.2fs, loss: %03.3f, miou: %03.3f, '
                'loss_g: %03.3f, loss_intra: %03.3f, loss_inter: %03.3f, loss_d: %03.3f',
                step, idx, len(data_loader), time.time() - batch_time, loss.cpu().item(), miou,
                loss_g.cpu().item(),
                loss_intra.cpu().item(),
                loss_inter.cpu().item(),
                loss_disc.cpu().item() if ['discriminative'] else 0
                )
        # break

    stamp_result = torch.cat([torch.Tensor(i) for i in stamp_result], dim=0).transpose(0, 1) / 128
    bias_see.generate_bias_ditributon(stamp_result, sample_size=128)

    logger.info('epoch [%03d]: elapsed time:%0.2fs, avg loss: %03.3f, miou: %03.3f, '
                'avg loss_g: %03.3f, avg loss_intra: %03.3f, avg loss_inter: %03.3f, avg loss_d: %03.3f,',
                step, time.time() - _start_time,
                accumulate_loss / len(data_loader), accumulate_iou / len(data_loader),
                accumulate_loss_g / len(data_loader),
                accumulate_loss_m1 / len(data_loader), accumulate_loss_m2 / len(data_loader),
                accumulate_loss_d / len(data_loader))

    logger.info('*' * 100)



    return accumulate_loss / len(data_loader)

def valid(model, data_loader, params, logger, step, saver, dataset, bias_g):
    model.eval()

    _start_time = time.time()
    accumulate_loss = 0
    accumulate_iou = 0
    accumulate_loss_g = 0
    accumulate_loss_m1 = 0
    accumulate_loss_m2 = 0
    accumulate_loss_d = 0
    pred_dict = {'version': 'V0',
                 'results': {},
                 'external_data': {
                     'used': True,
                     'details': 'provided i3D feature'
                 },
                 'params': params}

    logger.info('validing:' + '*' * 106)

    for idx, batch_data in enumerate(data_loader):
        batch_time = time.time()

        sent_list, sent_feat, sent_len, sent_mask, \
        video_duration, vid_list, \
        ori_video_feat, ori_nfeats, ori_video_mask, ori_gt, \
        pseudo_video_feat, pseudo_nfeats, pseudo_video_mask, pseudo_gt = perpare_data(batch_data)

        B, T, _ = ori_video_feat.size()

        with torch.no_grad():

            ori_span_prob, ori_match_prob, pseudo_match_prob, \
            ori_disc_prob, pseudo_disc_prob = model(
                sent_feat, sent_mask,
                ori_video_feat, ori_video_mask,
                pseudo_video_feat, pseudo_video_mask,
                ori_gt['temporal_labels'], ori_gt['fore_masks'], ori_gt['back_masks'],
                pseudo_gt['temporal_labels'], pseudo_gt['fore_masks'], pseudo_gt['back_masks'],
            )

            # LOSS
            # Grounding Loss
            loss_g = span_ground_loss(ori_span_prob['start'], ori_span_prob['end'], ori_gt['framestps'])

            # Cross-Modal Semantic Matching Loss
            #   m1: intra-video
            loss_intra = params['loss_m1_lambda'] * ( \
            BCE_loss(ori_match_prob, ori_gt['temporal_labels'], ori_video_mask) + \
            BCE_loss(pseudo_match_prob, pseudo_gt['temporal_labels'], pseudo_video_mask))

            #   m2: inter-videos
            ori_mask = ori_gt['temporal_labels']
            pseudo_mask = pseudo_gt['temporal_labels']
            ori_match_prob = masked_softmax(ori_match_prob, ori_mask.cuda())
            pseudo_match_prob = masked_softmax(pseudo_match_prob, pseudo_mask.cuda())

            loss_inter = params['loss_m2_lambda'] * matching_KL_divergence(
                ori_match_prob, pseudo_match_prob,
                ori_gt['framestps'], pseudo_gt['framestps']
            )

            # Temporal Order Discrimination Loss
            # loss_disc = temporal_order_discrimination_loss(ori_disc_prob, pseudo_disc_prob,
            #                                                criterion_domain)
            loss = loss_g + loss_intra + loss_inter #+ params['loss_disc_lambda'] * loss_disc
            # loss_ori = loss_g + loss_intra + loss_inter #+ params['loss_disc_lambda'] * loss_disc

            # print(vid_list, ori_video_feat.shape, ori_nfeats, ori_video_mask.sum(1), ori_gt['framestps'], sep='\n')
            # for xx in range(len(vid_list)):
            #     # assert ori_nfeats[xx] > ori_gt['framestps'][xx][1], (ori_nfeats[xx], ori_gt['framestps'][xx][1])
            #     if ori_nfeats[xx] <= ori_gt['framestps'][xx][1]:
            #         ori_gt['framestps'][xx][1] = ori_nfeats[xx].item() - 1
            # # print(vid_list, ori_video_feat.shape, ori_nfeats, ori_video_mask.sum(1), ori_gt['framestps'], sep='\n')
            #
            #
            # (new_feats, new_mask, new_mask_label, new_mask_fore, new_mask_back, new_sampled_points,
            #  aug_framestamps, aug_nfeats, aug_video_feature, new_mask_label_p, new_mask_fore_p,
            #  new_mask_back_p) = bias_g.sample_point(
            #     vid_list, ori_video_feat, ori_nfeats, ori_video_mask, ori_gt['framestps']
            # )
            #
            # # print(ori_gt['framestps'], new_sampled_points)
            # # for iii, xxx in enumerate(new_feats[0]):
            # #     print(iii, xxx[0], ori_video_feat[0][iii][0], new_mask[0][iii])
            # # exit()
            #
            # ori_video_feat = new_feats
            # ori_video_mask = new_mask
            # ori_gt['temporal_labels'] = new_mask_label
            # ori_gt['fore_masks'] = new_mask_fore
            # ori_gt['back_masks'] = new_mask_back
            # ori_gt['framestps'] = new_sampled_points
            #
            # # new_feats, new_mask, new_mask_label, new_mask_fore, new_mask_back, new_sampled_points, _, _, _, _, _, _ = bias_ps.sample_point(
            # #     vid_list, pseudo_video_feat, pseudo_nfeats, pseudo_video_mask
            # # )
            # pseudo_video_feat = aug_video_feature
            # pseudo_video_mask = new_mask
            # pseudo_gt['temporal_labels'] = new_mask_label_p
            # pseudo_gt['fore_masks'] = new_mask_fore_p
            # pseudo_gt['back_masks'] = new_mask_back_p
            # pseudo_gt['framestps'] = aug_framestamps
            #
            # # pseudo_video_feat = new_feats
            # # pseudo_video_mask = new_mask
            # # pseudo_gt['temporal_labels'] = new_mask_label
            # # pseudo_gt['fore_masks'] = new_mask_fore
            # # pseudo_gt['back_masks'] = new_mask_back
            # # pseudo_gt['framestps'] = new_sampled_points
            #
            # ori_span_prob_old = ori_span_prob
            #
            # ori_span_prob, ori_match_prob, pseudo_match_prob, \
            #     ori_disc_prob, pseudo_disc_prob = model(
            #     sent_feat, sent_mask,
            #     ori_video_feat, ori_video_mask,
            #     pseudo_video_feat, pseudo_video_mask,
            #     ori_gt['temporal_labels'], ori_gt['fore_masks'], ori_gt['back_masks'],
            #     pseudo_gt['temporal_labels'], pseudo_gt['fore_masks'], pseudo_gt['back_masks'],
            # )
            #
            # ori_span_prob['start'] = ori_span_prob_old['start'] + 0.01 * ori_span_prob['start']
            # ori_span_prob['end'] = ori_span_prob_old['end'] + 0.01 * ori_span_prob['end']
            #
            # # LOSS
            # # Grounding Loss
            # loss_g = span_ground_loss(ori_span_prob['start'], ori_span_prob['end'], ori_gt['framestps'])
            #
            # # Cross-Modal Semantic Matching Loss
            # #   m1: intra-video
            # loss_intra = params['loss_m1_lambda'] * ( \
            #             BCE_loss(ori_match_prob, ori_gt['temporal_labels'], ori_video_mask) + \
            #             BCE_loss(pseudo_match_prob, pseudo_gt['temporal_labels'], pseudo_video_mask))
            #
            # #   m2: inter-videos
            # ori_mask = ori_gt['temporal_labels']
            # pseudo_mask = pseudo_gt['temporal_labels']
            # ori_match_prob = masked_softmax(ori_match_prob, ori_mask.cuda())
            # pseudo_match_prob = masked_softmax(pseudo_match_prob, pseudo_mask.cuda())
            #
            # # ori和pseudo时间片段长度不一样的话，就不能计算这个loss
            # loss_inter = params['loss_m2_lambda'] * matching_KL_divergence(
            #     ori_match_prob, pseudo_match_prob,
            #     ori_gt['framestps'], pseudo_gt['framestps']
            # )
            # # loss_inter = torch.Tensor([0.]).to(ori_match_prob.device)
            #
            # # Temporal Order Discrimination Loss
            # # loss_disc = temporal_order_discrimination_loss(ori_disc_prob, pseudo_disc_prob,
            # #                                                criterion_domain)
            # loss = loss_g + loss_intra + loss_inter # + params['loss_disc_lambda'] * loss_disc
            #
            # loss = 0.01 * loss + loss_ori


            pred_time, score = span_pred(ori_span_prob['start'].cpu(), ori_span_prob['end'].cpu())
            pred_time = dataset.frame2sec(pred_time.float(), duration=video_duration, nfeats=ori_nfeats)
            miou = compute_mean_iou(pred_time.float().data, ori_gt['timestps'].data)

        accumulate_loss += loss.cpu().item()
        accumulate_iou += miou.cpu().item()
        accumulate_loss_g += loss_g.cpu().item()
        accumulate_loss_m1 += loss_intra.cpu().item()
        accumulate_loss_m2 += loss_inter.cpu().item()

        if params['batch_log_interval'] != -1 and idx % params['batch_log_interval'] == 0:
            logger.info('valid: epoch[%03d], batch[%04d/%04d], elapsed time=%0.2fs, loss: %03.3f, miou: %03.3f, '
                        'loss_g: %03.3f, loss_m1: %03.3f, loss_m2: %03.3f',
                        step, idx, len(data_loader), time.time() - batch_time, loss.cpu().item(), miou,
                        loss_g.cpu().item(), loss_intra.cpu().item(), loss_inter.cpu().item())

        # submits
        pred_time = pred_time.cpu().data.numpy()
        score = score.cpu().data.numpy()
        ts_time = ori_gt['timestps'].cpu().data.numpy()
        video_duration = video_duration.cpu().data.numpy()
        for idx in range(B):
            video_key = vid_list[idx]
            if video_key not in pred_dict['results']:
                pred_dict['results'][video_key] = list()
            pred_dict['results'][video_key].append({
                'sentence': sent_list[idx],
                'timestamp': pred_time[idx].tolist(),
                'gt_timestamp': ts_time[idx].tolist(),
                'score': score[idx].tolist(),
                'video_duration': video_duration[idx].tolist(),
            })

    saver.save_submits(pred_dict, step)
    logger.info('epoch [%03d]: elapsed time:%0.4fs, avg loss: %03.3f, miou: %03.3f '
                'avg loss_g: %03.3f, avg loss_m1: %03.3f, avg loss_m2: %03.3f',
                step, time.time() - _start_time,
                accumulate_loss / len(data_loader), accumulate_iou / len(data_loader),
                accumulate_loss_g / len(data_loader),
                accumulate_loss_m1 / len(data_loader), accumulate_loss_m2 / len(data_loader))
    logger.info('*' * 100)

    return accumulate_iou / len(data_loader)

def select_dataset_and_cfn(dataset_name):
    if dataset_name in ['charades', 'charades_cd']:
        from dataset.charades_pair_aug import collate_fn, CharadesVideoAugVideoPair
        data_class = CharadesVideoAugVideoPair
        cfn = collate_fn
    elif dataset_name in ['anet', 'anet_cd']:
        from dataset.anet_pair_aug import collate_fn, ANetVideoAugVideoPair
        data_class = ANetVideoAugVideoPair
        cfn = collate_fn
    else:
        assert False, 'Error datasetname' + dataset_name
    return data_class, cfn

def main(params):
    logging.basicConfig()
    logger = logging.getLogger(params['alias'])
    gpu_id = set_device(logger, params['gpu_id'])
    logger = logging.getLogger(params['alias'] + '(%d)' % gpu_id)
    set_device(logger, params['gpu_id'])
    logger.setLevel(logging.INFO)

    saver = ModelSaver(params, os.path.abspath('./third_party/densevid_eval'))
    model = constract_model(params, logger)
    model = torch.nn.DataParallel(model).cuda()

    data_class, train_cfn = select_dataset_and_cfn(params['train'])
    train_set = data_class(
        params['train_data'],
        params['train_featpath'],
        params,
        logger
    )
    train_loader = DataLoader(train_set, batch_size=params['batch_size'][0],
                              shuffle=True, num_workers=8, collate_fn=train_cfn)

    valid_data_class, valid_cfn = select_dataset_and_cfn(params['valid'])
    valid_set = valid_data_class(
        params['val_data'],
        params['valid_featpath'],
        params,
        logger
    )
    valid_loader = DataLoader(valid_set, batch_size=params['batch_size'][2],
                              shuffle=False, num_workers=4, collate_fn=valid_cfn)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if params['optim'].lower() in ['adam']:
        optimizer = torch.optim.Adam(parameters,
                                     lr=params['lr'],
                                     weight_decay=params['weight_decay'],
                                     eps=1e-6)
    elif params['optim'].lower() in ['adamw']:
        optimizer = torch.optim.AdamW(parameters,
                                      lr=params['lr'],
                                      weight_decay=params['weight_decay'])
    elif params['optim'].lower() in ['sgd']:
        optimizer = torch.optim.SGD(parameters,
                                    lr=params['lr'],
                                    weight_decay=params['weight_decay'],
                                    momentum=params['momentum'])

    if params['lr_schd'].lower() in ['multistep', 'ms']:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=params['lr_step'], gamma=params["lr_decay_rate"])
    elif params['lr_schd'].lower() in ['lambda', 'l']:
        lambda1 = lambda epoch: params['lr'] - epoch*1e-6
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=-1)

    criterion_domain = torch.nn.CrossEntropyLoss().cuda()


    bias_g = BiasDataGenerator(aug_fn=train_set.data_aug.aug_data, dataset_name='ori_video')
    bias_ps = BiasDataGenerator(aug_fn=train_set.data_aug.aug_data, dataset_name='pseudo_video')
    bias_val = BiasDataGenerator(dataset_name='val_video')
    bias_see = BiasDataGenerator(dataset_name='generated_time')

    coords = []
    coords_ps = []
    coords_val = []

    i = 0
    for idx, batch_data in tqdm(enumerate(train_loader), total=len(train_loader), desc='train bias'):
        i += 1
        if i > len(train_loader) / 10:
            break

        sent_list, sent_feat, sent_len, sent_mask, \
            video_duration, vid_list, \
            ori_video_feat, ori_nfeats, ori_video_mask, ori_gt, \
            pseudo_video_feat, pseudo_nfeats, pseudo_video_mask, pseudo_gt = perpare_data(batch_data)
        coords.append(ori_gt['timestps'])
        coords_ps.append(pseudo_gt['timestps'])

    i = 0
    for idx, batch_data in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='valid bias'):
        i += 1
        if i > len(valid_loader) / 10:
            break

        sent_list, sent_feat, sent_len, sent_mask, \
            video_duration, vid_list, \
            ori_video_feat, ori_nfeats, ori_video_mask, ori_gt, \
            pseudo_video_feat, pseudo_nfeats, pseudo_video_mask, pseudo_gt = perpare_data(batch_data)
        coords_val.append(ori_gt['timestps'])

    sample_size = ori_video_feat.shape[1]
    coords = torch.cat(coords, dim=0) / sample_size
    # coords = torch.cat(coords_val, dim=0) / 128
    coords_ps = torch.cat(coords_ps, dim=0) / sample_size
    # coords_ps = torch.cat(coords_val, dim=0) / 128
    coords_val = torch.cat(coords_val, dim=0) / sample_size

    bias_g.generate_bias_ditributon(
        coords=coords.T,
        sample_size=sample_size,
    )
    bias_ps.generate_bias_ditributon(
        coords=coords_ps.T,
        sample_size=sample_size,
    )
    bias_val.generate_bias_ditributon(
        coords=coords_val.T,
        sample_size=sample_size,
    )


    statistics = {'loss': {}, 'mIoU': {}}
    # valid(model, valid_loader, params, logger, 0, saver, valid_set)
    for step in range(params['epoch']):
        # if (step + 1) % 15 == 0:
        # if (step + 1) % 5 == 0:
        # if (step + 1) % 1 == 0:
        if step in [15, 20, 25, 30]:
            model.module.dynamic_expand()

        # train
        loss = train(model, train_loader, params, logger, step, optimizer, criterion_domain, train_set, bias_g, bias_ps, bias_see)
        # loss = train(model, valid_loader, params, logger, step, optimizer, criterion_domain, train_set)
        lr_scheduler.step()

        # record, validation and saving
        if (step + 1) % params['test_interval'] == 0 or step == 0:
            statistics['loss'][step] = round(loss, 3)
            LoggerInfo(logger, 'loss statistics:', statistics['loss'])
        if (step + 1) % params['test_interval'] == 0:
            mIoU = valid(model, valid_loader, params, logger, step, saver, valid_set, bias_g)
            statistics['mIoU'][step] = round(mIoU*100, 2)
            LoggerInfo(logger, 'mIoU statistics:', statistics['mIoU'])
        if (step + 1) % params['save_model_interval'] == 0 or (step + 1) == params['epoch']:
            save_path = saver.save_model_path(step)
            torch.save(model.module.state_dict(), save_path)
            logger.info('Save model in %s', save_path)

    StatisticsPrint(statistics, 'loss')
    StatisticsPrint(statistics, 'mIoU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False,
                        help='option to debug')

    # Datasets
    parser.add_argument('--feature_type', type=str, default='i3d',
                        help='feature type')
    parser.add_argument('--vfeat_fn', type=str, default='raw',
                        help='feature type')
    parser.add_argument('--cfg', type=str, default='charades_cd_i3d.yml',
                        help='domain adaptation configure')

    parser.add_argument('--train', type=str, default='charades',
                        help='source dataset')
    parser.add_argument('--valid', type=str, default='charades',
                        help='source dataset')

    parser.add_argument('--train_data', type=str, default='../../data/Charades/train.json',
                        help='source data path')
    parser.add_argument('--val_data', type=str, default='../../data/Charades/test.json',
                        help='validation data path')

    parser.add_argument('--train_featpath', type=str, default='../../data/Charades/charades_i3d_rgb.hdf5',
                        help='feature path')
    parser.add_argument('--valid_featpath', type=str, default='../../data/Charades/charades_i3d_rgb.hdf5',
                        help='feature path')

    parser.add_argument('--wordtoix_path', type=str, default='words/wordtoix.npy',
                        help='wordtoix_path')
    parser.add_argument('--ixtoword_path', type=str, default='words/ixtoword.npy',
                        help='ixtoword_path')
    parser.add_argument('--word_fts_path', type=str, default='words/word_glove_fts_init.npy',
                        help='word_fts_path')

    # Data_aug
    parser.add_argument('--if_aug', action='store_true', default=False,
                        help='data augment')
    parser.add_argument('--aug_percentage', type=float, default=0.5,
                        help='aug_percentage')
    parser.add_argument('--aug_mode', type=str, default='gt_translate',
                        help='checkpoint')

    # Load and Save
    parser.add_argument('--start_from', type=str, default=None,
                        help='checkpoint')

    # Interval
    parser.add_argument('--save_model_interval', type=int, default=1,
                        help='save the model parameters every a certain step')
    parser.add_argument('--batch_log_interval', type=int, default=50,
                        help='log interval')
    parser.add_argument('--batch_log_interval_test', type=int, default=50,
                        help='log interval')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='test interval between training')

    # Training Setting
    parser.add_argument('-b', '--batch_size', default=[32, 28, 64], type=int, nargs="+",
                        metavar='N', help='mini-batch size ([train, valid, test])')
    # parser.add_argument('--epoch', type=int, default=30,
    parser.add_argument('--epoch', type=int, default=60,
                        help='training epochs in total')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='used in data loader(only 1 is supported because of bugs in h5py)')
    parser.add_argument('--alias', type=str, default='test',
                        help='alias used in model/checkpoint saver')
    parser.add_argument('--runs', type=str, default='runs',
                        help='folder where models are saved')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='the id of gup used to train the model, -1 means automatically choose the best one')

    # Loss
    parser.add_argument('--loss_disc_lambda', type=float, default=1.0,
                        help='weight of loss_d in final loss')
    parser.add_argument('--loss_m1_lambda', type=float, default=1,
                        help='loss_da in final loss')
    parser.add_argument('--loss_m2_lambda', type=float, default=1,
                        help='loss_da in final loss')

    # Optim and Lr
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimizer')
    parser.add_argument('--lr_schd', type=str, default='ms',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate used to train the model')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay learning rate by this value every decay step')
    parser.add_argument('--lr_step', type=int, nargs='+', default=[15],
                        help='lr_steps used to decay the learning_rate')
    parser.add_argument('--momentum', type=float, default=0.8,
                        help='momentum used in the process of learning')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay, i.e. weight normalization')
    parser.add_argument('--grad_clip', action='store_true', default=False,
                        help='gradient clip threshold(not used)')
    parser.add_argument('--grad_clip_max', type=float, default=1.0,
                        help='gradient clip threshold(not used)')
    parser.add_argument('--group_weight', action='store_true', default=False,
                        help='group_weight')

    # Model
    parser.add_argument('--model', type=str, default="QAVE_match",
                        help='the model to be used')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='rnn_dropout')

    # Language
    parser.add_argument('--sent_encoder', type=str, default='rnn',
                        help='sent encoder')
    parser.add_argument('--sent_embedding_dim', type=int, default=300)
    parser.add_argument('--sent_rnn_hiddendim', type=int, default=256,
                        help='hidden dimension of rnn')
    parser.add_argument('--sent_rnn_layers', type=int, default=2,
                        help='layers number of rnn')
    parser.add_argument('--sent_rnn_cell', type=str, default='lstm',
                        help='rnn cell used in the model')
    parser.add_argument('--sent_len', type=int, default=20,
                        help='layers number of rnn')

    # Video
    parser.add_argument('--video_encoder', type=str, default='query_aware_encoder',
                        help='video encoder')
    parser.add_argument('--video_len', type=int, default=128,
                        help='vdieo len')
    parser.add_argument('--video_feature_dim', type=int, default=1024)
    parser.add_argument('--video_rnn_hiddendim', type=int, default=256,
                        help='hidden dimension of rnn')
    parser.add_argument('--video_rnn_layers', type=int, default=2,
                        help='layers number of rnn')
    parser.add_argument('--video_rnn_cell', type=str, default='lstm',
                        help='rnn cell used in the model')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='seq mask')

    # Cross-Modal Interaction
    parser.add_argument('--crossmodal', type=str, default='vs',
                        help='video-sent fusion manner')

    # Span Predictor
    parser.add_argument('--predictor', type=str, default="mlp",
                        help='the predictor to be used')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                        help='hidden dimension of mlp')
    parser.add_argument('--span_hidden_dim', type=int, default=128,
                        help='hidden dimension of rnn')


    # Matching setting
    parser.add_argument('--m_cross', type=str, default="concat",
                        help='')
    parser.add_argument('--m_temp', type=str, default="none",
                        help='')
    parser.add_argument('--m_pred', type=str, default="mlp",
                        help='')
    parser.add_argument('--m_pred_activ', type=str, default="relu",
                        help='')
    parser.add_argument('--m_pred_hidden', type=int, default=1024,
                        help='')

    params = parser.parse_args()
    params = vars(params)

    cfgs_file = params['cfg']
    cfgs_file = os.path.join('cfgs',cfgs_file)
    with open(cfgs_file, 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    update_values(options_yaml, params)
    # print(params)


    main(params)
    print('Training finished successfully!')