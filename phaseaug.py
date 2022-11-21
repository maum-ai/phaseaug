import torch
import torch.nn as nn
from math import pi, sqrt
from alias_free_torch.filter import LowPassFilter1d as LPF

class PhaseAug(nn.Module):
    def __init__(
        self, 
        nfft=1024, 
        hop=256, 
        use_filter=True,
        var=6.0,
        delta_max=2.0, 
        cutoff=0.05, 
        half_width=0.012, 
        kernel_size=128, 
        filter_padding='constant'
    ):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.var = var
        self.delta_max = delta_max
        self.use_filter = use_filter
        self.register_buffer('window', torch.hann_window(nfft))
        self.register_buffer('phi_ref', torch.arange(nfft // 2 + 1).unsqueeze(0) * 2 * pi / nfft)

        if use_filter:
            self.lpf = LPF(cutoff, half_width, kernel_size=kernel_size, padding_mode=filter_padding)
            def sample_phi(self, batch_size):
                mu = self.lpf(
                    sqrt(self.var) *
                    torch.randn([batch_size, 1, self.nfft // 2 + 1], device=self.phi_ref.device) +
                    self.delta_max * (2. * torch.rand([batch_size, 1, 1], device=self.phi_ref.device) - 1.)
                ).squeeze(1)
                phi = mu * self.phi_ref
                return phi   #[B,nfft//2+1]
        else:
            def sample_phi(self, batch_size):
                mu = (
                    sqrt(self.var) *
                    torch.randn([batch_size, self.nfft // 2 + 1], device=self.phi_ref.device) +
                    self.delta_max * (2. * torch.rand([batch_size, 1], device=self.phi_ref.device) - 1.)
                )
                phi = mu * self.phi_ref
                return phi   #[B,nfft//2+1]
        self.sample_phi = sample_phi

    # x: audio [B,1,T] -> [B,1,T]
    # phi: [B,nfft//2+1]
    # also possible for x :[B,C,T] but we did not generalize it.
    def forward(self, x, phi=None):
        x = x.squeeze(1)  #[B,t]
        X = torch.stft(
            x,
            self.nfft,
            self.hop,
            window=self.window,
            return_complex=False
        )  #[B,F,T,2]
        if phi is None:
            phi = self.sample_phi(self, X.shape[0])

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
            return_complex=False
        )
        return x_aug.unsqueeze(1)  #[B,1,t]

    # x: audio [B,1,T] -> [B,1,T]
    # phi: [B,nfft//2+1]
    def forward_sync(self, x, x_hat, phi=None):
        x = torch.cat([x, x_hat], dim=0).squeeze(1) #[2B,t]
        X = torch.stft(
            x,
            self.nfft,
            self.hop,
            window=self.window,
            return_complex=False
        )  #[2B,F,T,2]
        if phi is None:
            phi = self.sample_phi(self, X.shape[0] // 2)
        phi = torch.cat([phi, phi], dim=0)

        phi[:, 0] = 0. # we are multiplying phi_ref to mu, so it is always zero in our scheme
        phi = phi.unsqueeze(-1)  #[2B,F,1]
        phi_cos = phi.cos()
        phi_sin = phi.sin()
        rot_mat = torch.cat(
            [phi_cos, -phi_sin, phi_sin, phi_cos],  #[2B,F,2,2]
            dim=-1).view(-1, self.nfft // 2 + 1, 2, 2)
        # We did not mention that we multiplied rot_mat to "the left side of X"
        # Paper will be modified at rebuttal phase for clarity.
        X_aug = torch.einsum('bfij ,bftj->bfti', rot_mat, X)
        x_aug = torch.istft(
            X_aug,
            self.nfft,
            self.hop,
            window=self.window,
            return_complex=False
        )
        return x_aug.unsqueeze(1).split(x_aug.shape[0] // 2, dim=0)  #[B,1,t],[B,1,t]