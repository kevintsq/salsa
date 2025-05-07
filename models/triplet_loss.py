import torch
from torch import nn


class TripletLossHardNegMiningPlus(nn.Module):
    def __init__(self, margin=1.):
        """
        triplet loss with hard negative mining, four cases. inspired by http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf section3.3
        :param criterion: loss function
        :param f1: [lvl, B, C, H, W]
        :param f2: [lvl, B, C, H, W]
        :return:
            loss
        """
        super().__init__()
        self.criterion = nn.TripletMarginLoss(margin=margin, reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, f1, f2):
        anchor = f1
        anchor_negative = torch.roll(f1, shifts=1, dims=0)
        positive = f2
        negative = torch.roll(f2, shifts=1, dims=0)

        # select in-triplet hard negative, reference: section3.3
        with torch.no_grad():
            case1 = self.mse(anchor, negative)
            case2 = self.mse(positive, anchor_negative)
            case3 = self.mse(anchor, anchor_negative)
            case4 = self.mse(positive, negative)
            distance_list = torch.stack([case1, case2, case3, case4])
            loss_case = torch.argmin(distance_list)

        # perform anchor swap if necessary
        if loss_case == 0:
            loss = self.criterion(anchor, positive, negative)
        elif loss_case == 1:
            loss = self.criterion(positive, anchor, anchor_negative)
        elif loss_case == 2:
            loss = self.criterion(anchor, positive, anchor_negative)
        elif loss_case == 3:
            loss = self.criterion(positive, anchor, negative)
        else:
            raise NotImplementedError
        return loss
