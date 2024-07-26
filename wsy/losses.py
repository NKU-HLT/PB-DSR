"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """监督对比学习： https://arxiv.org/pdf/2004.11362.pdf.
    它还支持SimCLR中的无监督对比损失"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """计算模型的损失。如果“labels”和“mask”都为None，则退化为SimCLR无监督损失：
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...]. **BXTXF**
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
        elif labels is None and mask is None: # 如果没有label，相当于是只有自己是正样例
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # [97,1]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # 构成了一个标签一样则值为1，否则为0的矩阵
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # torch.unbind()移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片。
        # 上方本质是将出最后一个维度外的维度都合并拉直了
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # 这里感觉应该是在batch维度取，为什么是取F维度的第一个？好像又是对的，因为这里one的含义是只要第一个特征？
            anchor_count = 1 
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # 如果anchor等于正样例的话，相乘就变成了自己与自己进行内积？
            anchor_count = contrast_count # 这个取到的是时间维度的大小
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), # 根据此处计算，应该是将anchor与正样例相乘
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # 减去了最大值，看注释是为了数值稳定

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
        exp_logits = torch.exp(logits) * logits_mask # 此处得到的是指数分子
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # 这一步得到的就是inf了， 这一步不太理解？

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

####################################最大池化损失##############################################################
def padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Examples:
        >>> lengths = torch.tensor([2, 2, 3], dtype=torch.int32)
        >>> mask = padding_mask(lengths)
        >>> print(mask)
        tensor([[False, False,  True],
                [False, False,  True],
                [False, False, False]])
    """
    batch_size = lengths.size(0)
    max_len = int(lengths.max().item())
    seq = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    seq = seq.expand(batch_size, max_len)
    return seq >= lengths.unsqueeze(1)

def max_pooling_loss(logits: torch.Tensor, # torch.Size([108, 89, 15])
                     target: torch.Tensor, # torch.Size([108])
                     lengths: torch.Tensor, # torch.Size([108])
                     min_duration: int = 0):
    ''' Max-pooling loss
        对于关键字，选择后部最高的帧。当任何一个帧被触发时，该关键字就会被触发。
        对于none关键字，选择最硬的帧，即后面填充物最低（后面关键字最高）的帧。当未触发所有帧时，不触发关键字。
    Attributes:
        logits: (B, T, D), D is the number of keywords
        target: (B)
        lengths: (B)
        min_duration: min duration of the keyword
    Returns:
        (float): loss of current batch
        (float): accuracy of current batch
    '''
    mask = padding_mask(lengths)
    num_utts = logits.size(0)
    num_keywords = logits.size(2) # 15

    target = target.cpu()
    loss = 0.0
    for i in range(num_utts): # 这里的双重循环指的是对于每个样本，遍历所有关键字，如果目标就是对应关键字，则做最大池化，如果目标是其他关键词或者非关键字，则做最小池化
        for j in range(num_keywords): # 前面4个需要skip，最后一个也需要skip
            #-------wsy add-----------------
            if j in [0,1,2,3,14]: # 跳过 blank pad sos eos 和最后的-1
                """
                00:'<s>'
                01:'<pad>'
                02:'</s>'
                03:'<unk>'
                14:'-1'
                """
                continue
            #-----------------------------------
            # Add entropy loss CE = -(t * log(p) + (1 - t) * log(1 - p))
            if target[i] == j:
                # For the keyword, do max-polling
                prob = logits[i, :, j]
                m = mask[i].clone().detach()
                m[:min_duration] = True
                prob = prob.masked_fill(m, 0.0)
                prob = torch.clamp(prob, 1e-8, 1.0)
                max_prob = prob.max() # wsy：好像大概懂了，就是将区域中最后一帧的概率设置为该区域内最大的概率
                loss += -torch.log(max_prob)
            else:
                # For other keywords or filler, do min-polling
                prob = 1 - logits[i, :, j] # 这个对应的应该就是第i个样本的第j个关键词的概率
                prob = prob.masked_fill(mask[i], 1.0)
                prob = torch.clamp(prob, 1e-8, 1.0)
                min_prob = prob.min() # 同上
                loss += -torch.log(min_prob)
    loss = loss / num_utts

    # Compute accuracy of current batch
    mask = mask.unsqueeze(-1)
    logits = logits.masked_fill(mask, 0.0)
    max_logits, index = logits.max(1)
    num_correct = 0
    for i in range(num_utts):
        max_p, idx = max_logits[i].max(0)
        # Predict correct as the i'th keyword
        if max_p > 0.5 and idx == target[i]:
            num_correct += 1
        # Predict correct as the filler, filler id < 0
        if max_p < 0.5 and target[i] < 0:
            num_correct += 1
    acc = num_correct / num_utts
    # acc = 0.0
    #-wsy fix-----------
    # return loss, acc
    return loss
    #------------------
