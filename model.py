import torch
import torchaudio
from torch import nn
import torch.functional as F


class VADNoiseModelEnhancer(nn.Module):
    def __init__(
        self,
        spec_size = 241,
        input_samplerate = 16000,
        enhancer_size = 48,
        noise_model_size = 24,
        vad_model_size = 24,
        n_mels = 24,
        smoothing = 0.8,
        device='cpu'
    ):
        super().__init__()
        self.device = device      
        self.melscale_transform = torchaudio.functional.melscale_fbanks(
            spec_size,
            f_min = 0,
            f_max = input_samplerate / 2.0,
            n_mels = n_mels,
            sample_rate = input_samplerate,
            norm = 'slaney',
        )
        self.melscale_transform = self.melscale_transform.to(device)
        
        # YOUR CODE HERE
        self.vad_model_gru = nn.GRU(n_mels, vad_model_size)
        self.vad_model_fc = nn.Linear(vad_model_size, 1)
        
        self.noise_model = nn.GRU(n_mels + 1, noise_model_size)
        
        self.enhance_model_gru = nn.GRU(n_mels * 2 + 1, enhancer_size)
        self.enhance_model_linear = nn.Sequential(
            nn.Linear(enhancer_size, spec_size),
            nn.Sigmoid(),
        )

    def eval(self):
        self.eval_state = True
        return
    
    def train(self):
        self.eval_state = False
        return

    def forward(self, input_spec: torch.Tensor) -> torch.Tensor:
        input_features = torch.matmul(
            input_spec.abs()**2,
            self.melscale_transform
        )
        
        vad_out, _ = self.vad_model_gru(input_features)
        vad_out_lin_proj = self.vad_model_fc(vad_out)
        inputs_vad = torch.cat((input_features, vad_out_lin_proj), dim=-1)
        
        noise_out, _ = self.noise_model(inputs_vad)
        inputs_vad_noise = torch.cat((input_features, noise_out, vad_out_lin_proj), dim=-1)
        
        enhance_out, _ = self.enhance_model_gru(inputs_vad_noise)
        gains = self.enhance_model_linear(enhance_out)
        
        if self.eval_state:
            for k in range(gains.shape[-2]-1):
                gains[...,k+1,:] = (
                    (1-self.smoothing)*gains[...,k+1,:] +
                    self.smoothing    *gains[...,k,:] )
        
        estimated_spec = input_spec * gains
        return estimated_spec
    

class BLSTM(nn.Module):
    def __init__(
        self,
        spec_size=101,
        n_mels=10,
        lstm_hidden_size=425,
        lstm_layers=3,
        attention_heads=1,
        input_samplerate=16000,
        dropout=0.4,
        device='cpu',
        bidirectional=False,
    ): 
        super().__init__()
        self.device = device      
        self.melscale_transform = torchaudio.functional.melscale_fbanks(
            spec_size,
            f_min=0,
            f_max=input_samplerate / 2.0,
            n_mels=n_mels,
            sample_rate=input_samplerate,
            norm='slaney',
        )
        self.melscale_transform = self.melscale_transform.to(device)

        self.device = device
        self.spec_size = spec_size
        
        # YOUR CODE HERE
        self.fc1 = nn.Linear(in_features=n_mels, out_features=425)
        self.bn1 = nn.BatchNorm1d(num_features=425)
        self.tanh1 = nn.Tanh()

        self.lstm = nn.LSTM(input_size=425, hidden_size=lstm_hidden_size, num_layers=lstm_layers, bidirectional=bidirectional, dropout=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=425 + lstm_hidden_size * (2 ** int(bidirectional)), num_heads=attention_heads)
        self.fc2 = nn.Linear(in_features=425 + lstm_hidden_size * (2 ** int(bidirectional)), out_features=425)
        self.bn2 = nn.BatchNorm1d(num_features=425)
        self.relu1 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=425, out_features=425)
        self.bn3 = nn.BatchNorm1d(num_features=425)
        self.relu2 = nn.ReLU()

        self.fc4 = nn.Linear(in_features=425, out_features=spec_size)
        self.bn4 = nn.BatchNorm1d(num_features=spec_size)
        self.relu3 = nn.ReLU()

    def eval(self):
        self.eval_state = True
        return
    
    def train(self):
        self.eval_state = False
        return

    def forward(self, input_spec: torch.Tensor):
        input_features = torch.matmul(
            input_spec.abs()**2,
            self.melscale_transform
        )

        out = self.fc1(input_features)
        out = self.bn1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.tanh1(out)

        out_skip1 = out.clone()

        out, _ = self.lstm(out)
        out = torch.cat((out, out_skip1), dim=-1)
        out, _ = self.attention(out, out, out)
        
        out = self.fc2(out)
        out = self.bn2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.relu1(out)

        out = self.fc3(out)
        out = self.bn3(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.relu2(out)

        out = self.fc4(out)
        out = self.bn4(out.permute(0, 2, 1)).permute(0, 2, 1)
        gains = self.relu3(out)

        out = input_spec * gains
        return out
    

class CNNBlock(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels, 
        kernel_size=(2, 5), 
        stride=(1, 2), 
        padding=(0, 2), 
        is_decoder=False, 
        is_last=False
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding) if not is_decoder else\
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU() if not is_last else nn.Identity(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class CRN(nn.Module):
    def __init__(
        self,
        n_fft=101,
        gru_layers=1,
        device='cpu',
    ):
        super().__init__()
        self.eps = 1e-9
        self.device = device
        self.spec_size = n_fft // 2 + 1

        self.en_conv1 = CNNBlock(in_channels=1, out_channels=16)
        self.en_conv2 = CNNBlock(in_channels=16, out_channels=32)
        self.en_conv3 = CNNBlock(in_channels=32, out_channels=64)
        self.en_conv4 = CNNBlock(in_channels=64, out_channels=128)
        self.en_conv5 = CNNBlock(in_channels=128, out_channels=256)

        self.gru = nn.GRU(input_size=256 * 17, hidden_size=256 * 17, num_layers=gru_layers, bidirectional=False, batch_first=True)
        self.norm_and_act = nn.Sequential(
            nn.LayerNorm(256 * 17),
            nn.LeakyReLU(),
        )

        self.de_conv1 = CNNBlock(in_channels=256 + 256, out_channels=128, is_decoder=True)
        self.de_conv2 = CNNBlock(in_channels=128 + 128, out_channels=64, is_decoder=True)
        self.de_conv3 = CNNBlock(in_channels=64 + 64, out_channels=32, is_decoder=True)
        self.de_conv4 = CNNBlock(in_channels=32 + 32, out_channels=16, is_decoder=True)
        self.de_conv5 = CNNBlock(in_channels=16 + 16, out_channels=3, is_decoder=True, is_last=True)

        self.compress = nn.Linear(in_features=self.spec_size * 2 - 1, out_features=self.spec_size)

    def forward(self, input_spec: torch.Tensor):
        # Feature Engineering
        x_real = input_spec.real
        x_imag = input_spec.imag

        x = torch.cat((x_real, x_imag), dim=-1)
        x = x.unsqueeze(1)

        # Encoding
        en_out1 = self.en_conv1(x)
        en_out2 = self.en_conv2(en_out1)
        en_out3 = self.en_conv3(en_out2)
        en_out4 = self.en_conv4(en_out3)
        en_out5 = self.en_conv5(en_out4)

        # GRU
        out = en_out5.permute(0, 2, 1, 3)
        out = en_out5.contiguous().view(out.size(0), out.size(1), -1)
        gru_out, _ = self.gru(out)
        gru_out = self.norm_and_act(gru_out)
        gru_out = gru_out.permute(0, 2, 1).contiguous().view(*en_out5.shape)

        # Decoding
        de_out1 = self.de_conv1(torch.cat((gru_out, en_out5), dim=1))
        de_out2 = self.de_conv2(torch.cat((de_out1, en_out4), dim=1))
        de_out3 = self.de_conv3(torch.cat((de_out2, en_out3), dim=1))
        de_out4 = self.de_conv4(torch.cat((de_out3, en_out2), dim=1))
        de_out5 = self.de_conv5(torch.cat((de_out4, en_out1), dim=1))
        masks = self.compress(de_out5.permute(1, 0, 2, 3))

        # Complex Spectrogram Reconstruction
        x_magnitude = input_spec.abs()
        x_phase = input_spec.angle()

        mag_mask = torch.sigmoid(masks[0, :].squeeze(0))
        magnitude = mag_mask * x_magnitude

        real_phase_mask = torch.tanh(masks[1, :].squeeze(0))
        imag_phase_mask = torch.tanh(masks[2, :].squeeze(0))

        real_phase_mask = real_phase_mask / torch.sqrt(real_phase_mask ** 2 + imag_phase_mask ** 2 + self.eps)
        imag_phase_mask = imag_phase_mask / torch.sqrt(real_phase_mask ** 2 + imag_phase_mask ** 2 + self.eps)

        real_part = magnitude * (real_phase_mask * torch.cos(x_phase) - imag_phase_mask * torch.sin(x_phase))
        imag_part = magnitude * (real_phase_mask * torch.sin(x_phase) + imag_phase_mask * torch.cos(x_phase))

        out = torch.complex(real_part, imag_part)
        return out