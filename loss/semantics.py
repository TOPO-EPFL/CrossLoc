import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.io import safe_printout


class CrossEntropyLoss2d(nn.Module):
    # Cross entropy loss for semantic segmentation
    # Reference: https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/misc.py#L41
    def __init__(self, weight=None, reduction='none', ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def trim_semantic_label(raw_labels: np.ndarray):
    """
    Trim semantic labels for urbanscape and naturescape dataset.
    Raw class id     New class id   Content
    0                0              Sky
    1	             1              Unclassified and temporary objects
    2	             1              Ground
    3	             2              Low Vegetation
    6	             3              Buildings
    9	             4              Water
    17	             5              Bridge Deck
    """

    out_labels = raw_labels.copy()
    old_class_ls = [0, 1, 2, 3, 6, 9, 17]
    new_class_ls = [0, 1, 1, 2, 3, 4, 5]
    for old_class, new_class in zip(old_class_ls, new_class_ls):
        out_labels[raw_labels == old_class] = new_class

    assert np.min(out_labels) >= 0 and np.max(out_labels) <= 5
    return out_labels


def semantics_classification_loss(uncertainty, semantic_logits, uncertainty_map, gt_labels, criterion, reduction):
    """
    Calculate depth regression loss.
    Note: there is no nodata value for the semantic data. All pixels' classification is regularized.
    @param uncertainty          Flag for uncertainty loss.
    @param semantic_logits      [B, C, H, W] predicted logits
    @param uncertainty_map      [B, C, H, W] Uncertainty map tensor.
    @param gt_labels            [B, 1, H, W] ---> [B, 480, 720] by default w/o augmentation, 6 unique classes by default.
    @param criterion            Loss criterion. It must return non-aggregated result for each instance within batch.
    @param reduction            Method to post-process the mini-batch loss, 'mean' for mean and None for not aggregating
    @return loss                Regression loss value.
    @return num_valid_sc_rate   Rate of valid scene coordinates.
    """

    batch_size = semantic_logits.size(0)
    gt_labels = gt_labels.squeeze(1).long()  # [B, H, W]

    """check predicted semantics accuracy for various constraints"""
    class_prediction = torch.argmax(F.log_softmax(semantic_logits.clone().detach(), dim=1), dim=1)  # [B, H, W]
    valid_semantics = class_prediction == gt_labels  # [B, H, W]

    num_valid_semantics = valid_semantics.reshape(batch_size, -1).sum(dim=1).cpu().numpy()  # [B]
    num_pixels_batch = valid_semantics.numel()
    num_pixels_instance = valid_semantics[0].numel()

    # assemble loss
    loss = 0
    """assemble loss"""
    if uncertainty is None:
        loss_classification = criterion(semantic_logits, gt_labels)  # [B, H, W]
        loss += torch.sum(loss_classification.reshape(batch_size, -1), dim=1)  # [B]
    elif uncertainty == 'MLE':
        raise NotImplementedError
    else:
        raise NotImplementedError

    assert len(loss) == batch_size

    valid_pred_rate = num_valid_semantics.sum() / num_pixels_batch  # scalar

    if reduction is None:
        loss /= num_pixels_instance  # [B], each item is the mean over all pixels within one instance
    elif reduction == 'mean':
        loss = loss.sum()  # scalar, mean over each pixels within the batch
        loss /= num_pixels_batch
    else:
        raise NotImplementedError
    return loss, valid_pred_rate
