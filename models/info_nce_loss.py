import numpy as np
import torch
from torch import nn


class InfoNCELoss(nn.Module):
    def forward(self, C, I):
        C_audio = torch.log_softmax(C, dim=0)
        C_text = torch.log_softmax(C, dim=1)

        assert C_audio.shape[0] == C_audio.shape[1], \
            f'Audio Features Shape: {C_audio.shape} Sentence Features Shape: {C_text.shape}'
        assert C_text.shape[0] == C_text.shape[1]

        loss = -0.5 * (C_audio[I].mean() + C_text[I].mean())
        return loss
