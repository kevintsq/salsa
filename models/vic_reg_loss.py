import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(mat):
    n, m = mat.shape
    assert n == m
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Projector(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps
        self.projector1 = Projector(1024, 128)
        self.projector2 = Projector(1024, 128)

    def forward(self, z1, z2):
        z1 = self.projector1(z1)
        z2 = self.projector2(z2)

        # Invariance loss
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

        # Covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)

        cov_loss = off_diagonal(cov_z1).pow(2).sum() / z1.shape[1] + \
                   off_diagonal(cov_z2).pow(2).sum() / z2.shape[1]

        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss


class SimSiamDualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 使用相同结构 projector/predictor
        self.audio_projector = Projector()
        self.text_projector = Projector()

        self.audio_predictor = Projector(in_dim=128, hidden_dim=64, out_dim=128)
        self.text_predictor = Projector(in_dim=128, hidden_dim=64, out_dim=128)

    def forward(self, audio_input, text_input):
        # 得到基础表征
        z_a = self.audio_projector(audio_input)  # [B, 128]
        z_t = self.text_projector(text_input)    # [B, 128]

        # 预测器输出
        p_a = self.audio_predictor(z_a)  # [B, 128]
        p_t = self.text_predictor(z_t)

        return p_a, z_t.detach(), p_t, z_a.detach()


class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimSiamDualEncoder()

    def forward(self, a, t):
        p_a, z_t, p_t, z_a = self.encoder(a, t)

        # 计算相似度
        loss_a = -F.cosine_similarity(p_a, z_t).mean()
        loss_t = -F.cosine_similarity(p_t, z_a).mean()

        return 0.5 * (loss_a + loss_t)
