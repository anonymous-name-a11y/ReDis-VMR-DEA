import torch
import torch.nn as nn
from torch.nn import functional as F

from .networks.attention import *
from .components import SentenceEncoder, VideoEncoder, SpanPredictor, CrossModalInteraction, TemporalOrderDiscriminator
from .components.DistributionAlign import VideoTextSemanticMatch

class GMD(nn.Module):
    def __init__(self, video_seq_set, sent_seq_set, grounding_set, matching_set, logger, drop_out):
        super(GMD, self).__init__()

        # self.use_dynamic = False
        self.use_dynamic = True

        sent_encoder = SentenceEncoder.select_sent_encoder(sent_seq_set['name'], logger)
        video_encoder = VideoEncoder.select_video_encoder(video_seq_set['name'], logger)
        self.video_if_mask = video_seq_set['mask']
        if not self.use_dynamic:
            # Sentece Encoder
            self.sentence_encoder = sent_encoder(
                sent_seq_set, logger
            )
            self.textual_dim = self.sentence_encoder.textual_dim

            # Video Encoder
            video_seq_set['query_dim'] = self.textual_dim
            self.video_encoder = video_encoder(
                video_seq_set, logger
            )
            self.visual_dim = self.video_encoder.visual_dim

            # Grounding
            self.CMI = CrossModalInteraction.select_CMI(grounding_set['cross_name'], logger)(
                self.visual_dim,
                self.textual_dim
            )
            self.cross_dim = self.CMI.cross_dim()
        else:
            self.layer_use_num = 1
            # self.layer_use_num = 3
            # self.layer_use_num = 5
            # Sentece Encoder
            self.sentence_encoder = nn.ModuleList([sent_encoder(sent_seq_set, logger) for _ in range(5)])
            self.textual_dim = self.sentence_encoder[0].textual_dim

            # Video Encoder
            video_seq_set['query_dim'] = self.textual_dim
            self.video_encoder = nn.ModuleList([video_encoder(video_seq_set, logger) for _ in range(5)])
            self.visual_dim = self.video_encoder[0].visual_dim

            # Grounding
            self.CMI = nn.ModuleList([CrossModalInteraction.select_CMI(grounding_set['cross_name'], logger)(self.visual_dim, self.textual_dim) for _ in range(5)])
            self.cross_dim = self.CMI[0].cross_dim()

            self.dynamic_linear = nn.Linear(self.cross_dim * self.layer_use_num, self.cross_dim)
            for param in self.dynamic_linear.parameters():
                param.requires_grad = False
            self.dynamic_linear_pseudo = nn.Linear(self.visual_dim * self.layer_use_num, self.visual_dim)
            for param in self.dynamic_linear_pseudo.parameters():
                param.requires_grad = False
            self.dynamic_linear_video = nn.Linear(self.visual_dim * self.layer_use_num, self.visual_dim)
            for param in self.dynamic_linear_video.parameters():
                param.requires_grad = False
            self.dynamic_linear_sent = nn.Linear(self.textual_dim * self.layer_use_num, self.textual_dim)
            for param in self.dynamic_linear_sent.parameters():
                param.requires_grad = False



        self.span_predictor = SpanPredictor.SpanPredictor_Boundary(
            self.cross_dim,
            grounding_set,
            drop_out=drop_out,
            logger=logger,
        )

        # Cross-Modal Macthing
        matching_set['cross']['video_dim'] = self.visual_dim
        matching_set['cross']['query_dim'] = self.textual_dim
        self.csmm = VideoTextSemanticMatch(
            matching_set['cross'],
            matching_set['temporal'],
            matching_set['predict']
        )
        self.matching_dim = self.csmm.temporal_dim

        # Temporal Order Discriminator
        tod = TemporalOrderDiscriminator.select_temporal_order_discriminator(
            'moment_pooling',
            logger
        )
        self.tod = tod(self.visual_dim, logger)

    def dynamic_expand(self):
        if not self.use_dynamic or self.layer_use_num >= 5:
            return
        self.layer_use_num += 1

        ow, ob = self.dynamic_linear.weight.data, self.dynamic_linear.bias.data
        self.dynamic_linear = nn.Linear(self.cross_dim * self.layer_use_num, self.cross_dim).to(ow.device)
        self.dynamic_linear.weight.data[:, :self.cross_dim * (self.layer_use_num - 1)] = ow
        self.dynamic_linear.bias.data[:self.cross_dim * (self.layer_use_num - 1)] = ob
        for param in self.dynamic_linear.parameters():
            param.requires_grad = False

        ow, ob = self.dynamic_linear_pseudo.weight.data, self.dynamic_linear_pseudo.bias.data
        self.dynamic_linear_pseudo = nn.Linear(self.visual_dim * self.layer_use_num, self.visual_dim).to(ow.device)
        self.dynamic_linear_pseudo.weight.data[:, :self.visual_dim * (self.layer_use_num - 1)] = ow
        self.dynamic_linear_pseudo.bias.data[:self.visual_dim * (self.layer_use_num - 1)] = ob
        for param in self.dynamic_linear_pseudo.parameters():
            param.requires_grad = False

        ow, ob = self.dynamic_linear_video.weight.data, self.dynamic_linear_video.bias.data
        self.dynamic_linear_video = nn.Linear(self.visual_dim * self.layer_use_num, self.visual_dim).to(ow.device)
        self.dynamic_linear_video.weight.data[:, :self.visual_dim * (self.layer_use_num - 1)] = ow
        self.dynamic_linear_video.bias.data[:self.visual_dim * (self.layer_use_num - 1)] = ob
        for param in self.dynamic_linear_video.parameters():
            param.requires_grad = False

        ow, ob = self.dynamic_linear_sent.weight.data, self.dynamic_linear_sent.bias.data
        self.dynamic_linear_sent = nn.Linear(self.textual_dim * self.layer_use_num, self.textual_dim).to(ow.device)
        self.dynamic_linear_sent.weight.data[:, :self.textual_dim * (self.layer_use_num - 1)] = ow
        self.dynamic_linear_sent.bias.data[:self.textual_dim * (self.layer_use_num - 1)] = ob
        for param in self.dynamic_linear_sent.parameters():
            param.requires_grad = False

        self.sentence_encoder[self.layer_use_num - 1].load_state_dict(self.sentence_encoder[self.layer_use_num - 2].state_dict())
        self.video_encoder[self.layer_use_num - 1].load_state_dict(self.video_encoder[self.layer_use_num - 2].state_dict())
        self.CMI[self.layer_use_num - 1].load_state_dict(self.CMI[self.layer_use_num - 2].state_dict())

        for param in self.sentence_encoder[self.layer_use_num - 2].parameters():
            param.requires_grad = False
        for param in self.video_encoder[self.layer_use_num - 2].parameters():
            param.requires_grad = False
        for param in self.CMI[self.layer_use_num - 2].parameters():
            param.requires_grad = False

    def forward(self, query_feat, query_mask,
                ori_video_feat, ori_video_mask,
                pseudo_video_feat, pseudo_video_mask,
                ori_temporal_mask, ori_fore_mask, ori_back_mask,
                pseudo_temporal_mask, pseudo_fore_mask, pseudo_back_mask):
        _, N, D_q = query_feat.size()

        if not self.use_dynamic:
            # Natural Language Modality
            word_feat, sent_embed = self.sentence_encoder(query_feat)

            # Video Modality
            ori_frame_feat = self.video_encoder(ori_video_feat, word_feat)
            pseudo_frame_feat = self.video_encoder(pseudo_video_feat, word_feat)

            # Cross Modal
            ori_cross_feat = self.CMI(ori_frame_feat, word_feat, sent_embed)
        else:
            ori_cross_feats = []
            pseudo_frame_feats = []
            sent_embeds = []
            ori_video_feats = []
            for idx in range(self.layer_use_num):
                word_feat, sent_embed = self.sentence_encoder[idx](query_feat)
                ori_frame_feat = self.video_encoder[idx](ori_video_feat, word_feat)
                pseudo_frame_feat = self.video_encoder[idx](pseudo_video_feat, word_feat)
                ori_cross_feat = self.CMI[idx](ori_frame_feat, word_feat, sent_embed)

                ori_cross_feats.append(ori_cross_feat)
                pseudo_frame_feats.append(pseudo_frame_feat)
                sent_embeds.append(sent_embed)
                ori_video_feats.append(ori_frame_feat)


            # ori_cross_feat = self.dynamic_linear(torch.cat(ori_cross_feats, dim=-1))
            # pseudo_frame_feat = self.dynamic_linear_pseudo(torch.cat(pseudo_frame_feats, dim=-1))
            # sent_embed = self.dynamic_linear_sent(torch.cat(sent_embeds, dim=-1))
            # ori_frame_feat = self.dynamic_linear_video(torch.cat(ori_video_feats, dim=-1))

            ori_cross_feat = sum(ori_cross_feats) / len(ori_cross_feats)
            pseudo_frame_feat = sum(pseudo_frame_feats) / len(pseudo_frame_feats)
            sent_embed = sum(sent_embeds) / len(sent_embeds)
            ori_frame_feat = sum(ori_video_feats) / len(ori_video_feats)

        # Matching
        ori_match_prob, ori_matching_feat = self.csmm(
            ori_frame_feat, sent_embed, ori_video_mask
        )
        pseudo_match_prob, pseudo_matching_feat = self.csmm(
            pseudo_frame_feat, sent_embed, pseudo_video_mask
        )

        # Span Predictor
        ori_gated_feat = ori_match_prob.unsqueeze(dim=2) * ori_cross_feat
        start_prob, end_prob = self.span_predictor(
            ori_gated_feat,
            v_mask=ori_video_mask if self.video_if_mask else None,
        )
        span_prob = {}
        span_prob['start'] = start_prob
        span_prob['end'] = end_prob

        # Temporal Order Discriminator
        ori_disc_prob = self.tod(ori_frame_feat, ori_temporal_mask, ori_fore_mask, ori_back_mask)
        pseudo_disc_prob = self.tod(pseudo_frame_feat, pseudo_temporal_mask, pseudo_fore_mask, pseudo_back_mask)

        return span_prob, ori_match_prob, pseudo_match_prob, \
                ori_disc_prob, pseudo_disc_prob

    def eval_forward(self, video_feat, query_feat, video_mask=None, sent_mask=None):
        _, N, D_q = query_feat.size()

        # Natural Language Modality
        word_feat, sent_embed = self.sentence_encoder(query_feat)

        # Video Modality
        frame_feat = self.video_encoder(video_feat, word_feat)

        # Cross Modal
        cross_feat = self.CMI(frame_feat, word_feat, sent_embed)

        # Matching
        match_prob, matching_feat = self.csmm(
            frame_feat, sent_embed, video_mask
        )

        # Span Predictor
        gated_feat = match_prob.unsqueeze(dim=2) * cross_feat
        start_prob, end_prob = self.span_predictor(
            gated_feat,
            v_mask=video_mask if self.video_if_mask else None,
        )
        span_prob = {}
        span_prob['start'] = start_prob
        span_prob['end'] = end_prob

        return span_prob
