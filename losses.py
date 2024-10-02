import torch
import torchaudio
import torch.nn.functional as F


class MelLoss(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        n_stft=101,
        n_mels=10,
        device='cpu'
    ):
        super().__init__()
        self.melscale_transform = torchaudio.functional.melscale_fbanks(
            n_stft,
            f_min = 0,
            f_max = sample_rate / 2.0,
            n_mels = n_mels,
            sample_rate = sample_rate,
            norm = 'slaney',
        ).to(device)

    def __str__(self):
        return "MelLoss"
        
    def forward(self, estimated_spec, reference_spec):        
        mel_error_spec = torch.matmul(            
            (estimated_spec - reference_spec).abs()**2,
            self.melscale_transform
        )
        return mel_error_spec.clamp(min=1e-6).log().mean(dim=-2).mean()
    

class dBLoss(torch.nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        
    def __str__(self):
        return "dBLoss"
        
    def forward(self, estimated_spec, reference_spec):        
        error_spectrogram = estimated_spec.abs() - reference_spec.abs()
        return error_spectrogram.pow(2).clamp(min=1e-6).log().mean(dim=-2).mean()
    

class GainPowLoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        
    def forward(self, estimated_gains, oracle_gains):        
        gain_error = (estimated_gains - oracle_gains).abs().pow(2)
        return gain_error.pow(2).clamp(min=1e-6).mean(dim=-2).mean()
    

class CombinedSpectralLoss(torch.nn.Module):
    def __init__(self, alpha = 0.5):
        super().__init__()
        self.mse_loss = torch.nn.L1Loss()
        self.alpha = alpha
    
    def forward(self, est_spec, clean_spec):
        est_spec_real = est_spec.real
        est_spec_imag = est_spec.imag
        
        clean_spec_real = clean_spec.real
        clean_spec_imag = clean_spec.imag
        
        l_mag = self.mse_loss(est_spec.abs(), clean_spec.abs())
        print(l_mag)
        l_ri = self.mse_loss(est_spec_real, clean_spec_real) + self.mse_loss(est_spec_imag, clean_spec_imag) 
        print(l_ri)
        loss = self.alpha * l_mag + (1 - self.alpha) * l_ri
        return loss.clamp(min=1e-6).mean()