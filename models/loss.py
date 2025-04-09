import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from convs.partpointnet import l2_normalize


class PPC(nn.Module, ABC):
    def __init__(self):
        super(PPC, self).__init__()

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = torch.div(contrast_logits, 0.1)
        loss_ppc = F.cross_entropy(contrast_logits.view(-1, contrast_logits.shape[-1]) , contrast_target.long())
        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self):
        super(PPD, self).__init__()

    def forward(self, contrast_logits, contrast_target):
        logits = torch.gather(contrast_logits.view(-1, contrast_logits.shape[-1]), 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()
        return loss_ppd


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs=30, nepochs=250,
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.01):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))


    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output.reshape(-1, student_output.shape[-1])
        # teacher centering and sharpening
        # temp = self.teacher_temp_schedule[epoch]
        teacher_output = teacher_output.reshape(-1, teacher_output.shape[-1])
        teacher_output = F.softmax(teacher_output / self.student_temp, dim=-1)
        # teacher_out = teacher_output.detach()
        loss = torch.sum(-teacher_output * F.log_softmax(student_out, dim=-1), dim=-1).mean()

        return loss


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) #将mask换成权重

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def label_neg(self, features, neg_features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)         # N,N   N,N+K

        neg_dot_contrast = torch.div(
            torch.matmul(anchor_feature, neg_features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(torch.cat([anchor_dot_contrast, neg_dot_contrast], dim=1), dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        neg_dot_contrast = neg_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(torch.cat([logits * logits_mask, neg_dot_contrast], dim=1))  #neg

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 将mask换成权重

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CatProtoDiff(nn.Module):
    def __init__(self, dist=0.1):
        super(CatProtoDiff, self).__init__()
        self.dis = dist

    def forward(self, cat_prototype):
        part_diff = torch.einsum('mc,nc->mn', cat_prototype, cat_prototype) - torch.eye(cat_prototype.shape[0], cat_prototype.shape[0]).cuda() - self.dis
        loss_diff = torch.sum(part_diff[part_diff > 0])
        return loss_diff


def Consistency(feats, att):
    B, N, K = att.shape
    consistency_loss = 0.0
    feats = feats.reshape(B * N, -1)
    att = att.reshape(B * N, -1)
    for i in range(K):
        indices = ((att.argmax(1) == i) & (att.max(1)[0] > 0.7))
        need = feats[indices]
        if indices.sum() > 0:
            consistency_loss += F.cosine_similarity(need[None, :, :], need[:, None, :], dim=-1).mean()
    return 1 - consistency_loss / K


def Distinctiveness(feats, att):
    B, N, K = att.shape
    records = []
    feats = feats.reshape(B * N, -1)
    att = att.reshape(B * N, -1)
    for i in range(K):
        indices = (att.argmax(1) == i)
        need = feats[indices]
        if indices.sum() > 0: 
            records.append(torch.mean(need, dim=0, keepdim=True))
    records = torch.cat(records, dim=0)
    distinctive_loss = F.cosine_similarity(records[None, :, :], records[:, None, :], dim=-1).mean()
    return distinctive_loss


class PGD_Prototype_Novel(nn.Module, ABC):
    def __init__(self):
        super(PGD_Prototype_Novel, self).__init__()
        self.seg_criterion = nn.CrossEntropyLoss()
        self.part_score_criterion = SupConLoss()

    def forward(self, preds, targets, predsbar, mode):
        loss_list = {}
        logits = preds['logits']
        logits = torch.div(logits, 0.1)
        logits_bar = predsbar['logits']
        logits_bar = torch.div(logits_bar, 0.1)
        part_scores = preds['part_scores']
        part_scores_bar = predsbar['part_scores']

        loss_ce = self.seg_criterion(torch.vstack([logits, logits_bar]), torch.cat([targets.long(), targets.long()]))
        loss_sc = self.part_score_criterion(l2_normalize(torch.stack([part_scores, part_scores_bar], dim=1)), targets.long())
        loss_con = (Consistency(preds['base_feats'], preds['part_logits']) + Consistency(predsbar['base_feats'], predsbar['part_logits'])) / 2
        loss_dis = (Distinctiveness(preds['base_feats'], preds['part_logits']) + Distinctiveness(predsbar['base_feats'], predsbar['part_logits'])) / 2
        if mode == 0:
            loss = loss_ce + loss_sc * 0.3 + loss_con * 0.1 + loss_dis * 0.1
            return logits, loss, loss_list
        else:
            loss = loss_ce
            return logits, loss, loss_list

