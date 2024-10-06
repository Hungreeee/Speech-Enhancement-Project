import time
import zipfile
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import os
import torch.multiprocessing as mp
import scipy
from torch.utils.tensorboard import SummaryWriter



class NoisySpeech(Dataset):
    def __init__(self, clean_zip_path, noisy_zip_path, extract_dir, device='cpu'):
        self.device = device
        self.extract_dir = extract_dir
        
        clean_dir = extract_dir + "/clean"
        noisy_dir = extract_dir + "/noisy"
        
        # Extract the ZIP files only if the directories don't exist or are empty
        if not os.path.exists(clean_dir) or not os.listdir(clean_dir):
            print(f"Extracting clean files to {clean_dir}...")
            with zipfile.ZipFile(clean_zip_path, 'r') as clean_zip:
                clean_zip.extractall(clean_dir)

        if not os.path.exists(noisy_dir) or not os.listdir(noisy_dir):
            print(f"Extracting noisy files to {noisy_dir}...")
            with zipfile.ZipFile(noisy_zip_path, 'r') as noisy_zip:
                noisy_zip.extractall(noisy_dir)
        
        # Find the actual subdirectories where the .wav files are stored
        clean_subdir = os.path.join(clean_dir, os.listdir(clean_dir)[0])  # Gets the first directory inside 'clean'
        noisy_subdir = os.path.join(noisy_dir, os.listdir(noisy_dir)[0])  # Gets the first directory inside 'noisy'
        
        # List all .wav files from the subdirectory
        self.__clean_wav_list__ = sorted([os.path.join(clean_subdir, f) 
                                          for f in os.listdir(clean_subdir) if f.endswith(".wav")])
        self.__noisy_wav_list__ = sorted([os.path.join(noisy_subdir, f) 
                                          for f in os.listdir(noisy_subdir) if f.endswith(".wav")])
        
        # Debugging: Print the number of files found
        print(f"Found {len(self.__clean_wav_list__)} clean .wav files")
        print(f"Found {len(self.__noisy_wav_list__)} noisy .wav files")

        # Ensure that both lists have the same number of files, or use the smaller length
        self.dataset_length = min(len(self.__clean_wav_list__), len(self.__noisy_wav_list__))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if idx >= self.dataset_length:
            raise IndexError("Index out of range for dataset")

        # Directly read the extracted files
        sr, np_clean_audio = scipy.io.wavfile.read(self.__clean_wav_list__[idx])
        sr, np_noisy_audio = scipy.io.wavfile.read(self.__noisy_wav_list__[idx])
        
        return torch.tensor(np_noisy_audio).to(self.device), torch.tensor(np_clean_audio).to(self.device), torch.tensor(sr)


def CollateNoisySpeech(itemlist):
    buffer_len = 6 * 48000  # Maximum length is 60 sec at 48kHz
    sample_len = min(min(len(noisy) for noisy, _, _ in itemlist), buffer_len)
    
    noisy_batch = torch.zeros((len(itemlist), buffer_len))
    clean_batch = torch.zeros((len(itemlist), buffer_len))
    
    for i, (noisy, clean, sr) in enumerate(itemlist):
        noisy_batch[i, :sample_len] = noisy[:sample_len]
        clean_batch[i, :sample_len] = clean[:sample_len]
    
    return noisy_batch[:, :sample_len], clean_batch[:, :sample_len], sr


class PreProcessor(torch.nn.Module):
    def __init__(
        self,
        input_samplerate=16000,
        resample_samplerate=16000,
        n_fft=480,
        power=None,
    ):
        super().__init__()
        self.output_size = n_fft // 2 + 1
        self.resample = transforms.Resample(input_samplerate, resample_samplerate)
        self.transform = transforms.Spectrogram(n_fft=n_fft, power=power, normalized=True, window_fn=torch.hann_window)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)
        spec = self.transform(resampled)
        spec = spec.permute(0, 2, 1)
        return spec


class PostProcessor(torch.nn.Module):
    def __init__(
        self,
        output_samplerate = 16000,
        resample_samplerate = 16000,
        n_fft = 480
    ):
        super().__init__()
        n_fft = n_fft
        self.resample = transforms.Resample(resample_samplerate, output_samplerate)
        self.transform = transforms.InverseSpectrogram(n_fft=n_fft, normalized=True, window_fn=torch.hann_window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.permute(0, 2, 1)
        waveform = self.transform(spec)
        resampled = self.resample(waveform)
        return resampled


class CombinedSpectralLoss(torch.nn.Module):
    def __init__(self, alpha = 0.5):
        super().__init__()
        self.mse_loss = nn.L1Loss()
        self.alpha = alpha
    
    def forward(self, est_spec, clean_spec):
        est_spec_real = est_spec.real
        est_spec_imag = est_spec.imag
        
        clean_spec_real = clean_spec.real
        clean_spec_imag = clean_spec.imag
        
        l_mag = self.mse_loss(est_spec.abs(), clean_spec.abs())
        l_ri = self.mse_loss(est_spec_real, clean_spec_real) + self.mse_loss(est_spec_imag, clean_spec_imag) 
        print(l_mag)
        print(l_ri)
        loss = self.alpha * l_mag + (1 - self.alpha) * l_ri
        return loss.clamp(min=1e-6).mean()


def train(
    dataloader, 
    dataset, 
    model, 
    preprocessor, 
    loss_fn, 
    optimizer, 
    writer,
    scheduler=None, 
    epochs=1,
    device='cuda',
    save_path='.runs/',
):
    size = len(dataset)
    model.train()
    start_time = time.perf_counter()

    for epoch in range(epochs):
        for batch, (noisy_batch, clean_batch, _) in enumerate(dataloader):
            noisy_spec = preprocessor(noisy_batch).to(device)
            clean_spec = preprocessor(clean_batch).to(device)

            batch_size = noisy_batch.shape[0]

            est_clean_spec, _ = model(noisy_spec)
            loss = loss_fn(est_clean_spec, clean_spec)
            writer.add_scalar("Loss/Step", loss, batch)
            if scheduler:
                scheduler.step(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (batch+1) % 10 == 0:
                torch.save(model, f"{save_path}/crn-model-checkpoints.pt")

            if (batch+1) % 10 == 0:
                curr_time = time.perf_counter()
                loss, current = loss.item(), 1 + (batch)*batch_size + epoch*size
                print(f"loss: {loss:>7f} [{current:>5d}/{size*epochs:>5d}] at {curr_time-start_time:>5f} sec")
                start_time = curr_time


class EncoderBlock(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels, 
        kernel_size=(2, 5), 
        stride=(1, 2), 
        padding=(0, 2), 
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out


class DecoderBlock(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels, 
        kernel_size=(2, 5), 
        stride=(1, 2), 
        padding=(0, 2), 
        is_last=False
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU() if not is_last else nn.Identity()
        )

    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out


class CRN(nn.Module):
    def __init__(
        self,
        n_fft=512,
        gru_layers=3,
        device='cuda',
    ):
        super().__init__()
        self.eps = 1e-9
        self.device = device
        self.spec_size = n_fft // 2 + 1

        self.en_conv1 = EncoderBlock(in_channels=1, out_channels=16)
        self.en_conv2 = EncoderBlock(in_channels=16, out_channels=32)
        self.en_conv3 = EncoderBlock(in_channels=32, out_channels=64)
        self.en_conv4 = EncoderBlock(in_channels=64, out_channels=128)
        self.en_conv5 = EncoderBlock(in_channels=128, out_channels=256)

        self.gru = nn.GRU(input_size=256 * 17, hidden_size=256 * 17, num_layers=gru_layers, bidirectional=False, batch_first=True)
        self.norm_and_act = nn.Sequential(
            nn.LayerNorm(256 * 17),
            nn.LeakyReLU(),
        )

        self.de_conv1 = DecoderBlock(in_channels=256 + 256, out_channels=128)
        self.de_conv2 = DecoderBlock(in_channels=128 + 128, out_channels=64)
        self.de_conv3 = DecoderBlock(in_channels=64 + 64, out_channels=32)
        self.de_conv4 = DecoderBlock(in_channels=32 + 32, out_channels=16)
        self.de_conv5 = DecoderBlock(in_channels=16 + 16, out_channels=3, is_last=True)

        self.out_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))
        self.attention = nn.MultiheadAttention(embed_dim=257, num_heads=1, batch_first=True)

    def calculate_gains(self, noisy_spec: torch.Tensor, clean_spec: torch.Tensor):
        return (clean_spec.abs() / noisy_spec.abs().clamp(min=1e-6)).clamp(max=1)

    def forward(self, input_spec: torch.Tensor):
        # Feature engineering
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

        # Obtain masks
        out = self.out_conv(de_out5)
        masks = out.permute(1, 0, 2, 3)

        # Complex spectrogram reconstruction
        x_magnitude = input_spec.abs()
        x_phase = input_spec.angle()

        mag_mask = masks[0, :].squeeze(0)
        magnitude = torch.sigmoid(mag_mask) * x_magnitude

        real_phase_mask = torch.tanh(self.attention(masks[1, :].squeeze(0), mag_mask, masks[1, :].squeeze(0))[0])
        imag_phase_mask = torch.tanh(self.attention(masks[2, :].squeeze(0), mag_mask, masks[1, :].squeeze(0))[0])

        real_phase_mask = real_phase_mask / torch.sqrt(real_phase_mask ** 2 + imag_phase_mask ** 2 + self.eps)
        imag_phase_mask = imag_phase_mask / torch.sqrt(real_phase_mask ** 2 + imag_phase_mask ** 2 + self.eps)

        real_part = magnitude * (real_phase_mask * torch.cos(x_phase) - imag_phase_mask * torch.sin(x_phase))
        imag_part = magnitude * (real_phase_mask * torch.sin(x_phase) + imag_phase_mask * torch.cos(x_phase))

        est_spec = torch.complex(real_part, imag_part)
        gains = self.calculate_gains(input_spec, est_spec)
        
        return est_spec, gains


def main():
    # mp.set_start_method('spawn')
    writer = SummaryWriter(log_dir="/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/tensorboard")

    SAVE_PATH = "/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/models/"

    N_FFT = 512
    N_MELS = 50
    GRU_LAYERS = 2
    RESAMPLE_SAMPLERATE = 16000
    ALPHA = 0.4

    LR = 1e-3
    EPOCHS = 2
    BATCH_SIZE = 32

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)

    clean_zip = "/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/clean_trainset_28spk_wav.zip"
    noisy_zip = "/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/noisy_trainset_28spk_wav.zip"
    extracted_path = "/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/extracted"
    # Load data, create dataloaders, set parameters
    dataset = NoisySpeech(clean_zip, noisy_zip, extracted_path, device=DEVICE)
    _, _, input_samplerate = dataset.__getitem__(0)

    enhancer = CRN(n_fft=N_FFT, gru_layers=GRU_LAYERS)
    enhancer.to(DEVICE)

    optimizer = torch.optim.RMSprop(params=enhancer.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=100, threshold=5e-2, min_lr=5e-5)
    loss = CombinedSpectralLoss(alpha=ALPHA)
    preprocessor = PreProcessor(input_samplerate=input_samplerate, n_fft=N_FFT, power=None)

    train_size = int(0.7 * len(dataset))
    eval_size = int(0.1 * len(dataset))
    leftover = len(dataset) - train_size - eval_size

    train_dataset, eval_dataset, _ = torch.utils.data.random_split(dataset, [train_size, eval_size, leftover])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=CollateNoisySpeech)
    
    print("Start Training...")
    train(
        dataloader=train_dataloader,
        dataset=train_dataset,
        model=enhancer,
        preprocessor=preprocessor,
        loss_fn=loss,
        optimizer=optimizer,
        writer=writer,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=DEVICE,
        save_path=SAVE_PATH,
    )
    print("Finish Training...")
    writer.flush()
    writer.close()
    torch.save(enhancer.state_dict(), SAVE_PATH + "crn-nfft_512-alp_04-comspecloss-bs_16.pt")
    print("Save Model...")


if __name__=="__main__":
    main()
