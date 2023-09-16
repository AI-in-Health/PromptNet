import pdb

import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def compute_loss(output, reports_ids, reports_masks,loss_ce):
    #criterion = LanguageModelCriterion()
    #loss = criterion(output, reports_ids, reports_masks).mean()
    reports_ids = reports_ids.flatten()
    reports_masks = reports_masks.flatten()  # Flatten the mask
    logits = output.reshape(-1, output.shape[-1])
    loss = loss_ce(logits, reports_ids)

    masked_loss = loss * reports_masks
    # Normalize the loss by the number of unmasked elements
    loss_token = masked_loss.sum() / reports_masks.sum()

    return loss_token
