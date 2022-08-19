import pdb

import torch
import torch.nn as nn


class SuperConLoss(nn.Module):

    def __init__(self, temperature=0.07, force_normalization=True):
        super(SuperConLoss, self).__init__()
        self.temperature = temperature
        self.force_normalization = force_normalization
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, anchor_feat, pos_feat, neg_feat):
        """
        Cross entropy classification loss for the query & positive/negative keys.
        :param anchor_feat: [1, D]
        :param pos_feat: [P, D]
        :param neg_feat: [N, D]
        :return: supervised contrastive learning loss
        """

        # init
        P = pos_feat.size(0)  # number of positives
        contrast_feat = torch.cat([anchor_feat, pos_feat, neg_feat], dim=0)  # [B, D], N=1+P+N
        if self.force_normalization:
            contrast_feat = nn.functional.normalize(contrast_feat, p=2, dim=-1)

        # compute logits table for any two feature vectors
        logits_table = torch.div(torch.matmul(contrast_feat, contrast_feat.T), self.temperature)  # [B, B]
        logits_max, _ = torch.max(logits_table, dim=1, keepdim=True)  # [B, 1]
        logits_table = logits_table - logits_max.detach()  # [B, B]

        # assume that any two samples from (anchor, positives) form a positive pair
        mask_positive_pairs = torch.zeros_like(logits_table)  # [B, B]
        mask_positive_pairs[:1+P, :1+P] = 1
        mask_positive_pairs[:1+P, :1+P] -= torch.eye(1+P).cuda()  # disable self-contrast pairs

        # mask for NCE loss denominator
        mask_denominator = torch.ones_like(logits_table)  # [B, B]
        mask_denominator -= torch.eye(logits_table.size(0)).cuda()  # [B, B], remove self-contrast

        # aggregate NCE loss
        exp_logits = torch.exp(logits_table) * mask_denominator
        log_prob = logits_table - torch.log(exp_logits.sum(dim=1, keepdims=True))  # [B, B]
        log_prob = (log_prob * mask_positive_pairs).sum(dim=1) / mask_positive_pairs.sum(dim=1).clamp(min=1)  # [B]
        loss = -log_prob.sum()  # scalar

        return loss
