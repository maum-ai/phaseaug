import torch
import torch.nn as nn
from math import pi


class PhaseAug(nn.Module):

    def __init__(self, nfft=1024, hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft))

    #x: audio [B,1,T] -> [B,1,T]
    #phi: [B, nfft//2+1]
    # also possible for x :[B,C,T] but we did not generalize it.
    def forward(self, x, phi=None):
        x = x.squeeze(1)  #[B,t]
        X = torch.stft(
            x,
            self.nfft,
            self.hop,
            window=self.window,
        )  #[B,F,T,2]
        if phi is None:
            phi = 2 * pi * torch.rand([X.shape[0], X.shape[1]], device=x.device)
        phi[:, 0] = 0. # we are multiplying phi_ref to mu, so it is always zero in our scheme
        phi = phi.unsqueeze(-1)  #[B,F,1]
        phi_cos = phi.cos()
        phi_sin = phi.sin()
        rot_mat = torch.cat(
            [phi_cos, -phi_sin, phi_sin, phi_cos],  #[B,F,2,2]
            dim=-1).view(-1, self.nfft // 2 + 1, 2, 2)
        # We did not mention that we multiplied rot_mat to "the left side of X"
        # Paper will be modified at rebuttal phase for clarity.
        X_aug = torch.einsum('bfij ,bftj->bfti', rot_mat, X)
        x_aug = torch.istft(
            X_aug,
            self.nfft,
            self.hop,
            window=self.window,
        )
        return x_aug.unsqueeze(1)  #[B,1,t]
