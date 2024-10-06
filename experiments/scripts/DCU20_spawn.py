import zipfile
import time

import torch
from torch import nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import torch.multiprocessing as mp
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


class PreProcessing(torch.nn.Module):
    def __init__(
        self,
        input_samplerate    = 16000,
        resample_samplerate = 16000,
        window_length_ms    = 30,
        device='cuda'  # Add device parameter
    ):
        super().__init__()
        self.device = device
        self.resample = torchaudio.transforms.Resample(orig_freq=input_samplerate, new_freq=resample_samplerate).to(self.device)
        n_fft = (2*window_length_ms * resample_samplerate) // 2000
        hop_length = n_fft // 2
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft,power=None,hop_length=hop_length).to(self.device)
        self.output_size = (n_fft+2)//2
        

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(self.device)
        # Resample the input
        # Convert to power spectrogram
        resampled = self.resample(waveform)
        spec = self.spec(resampled)
        
        return spec

        
class PostProcessing(torch.nn.Module):
    def __init__(
        self,
        output_samplerate   = 16000,
        resample_samplerate = 16000,
        window_length_ms    = 30,
        device='cuda'  # Add device parameter
    ):
        super().__init__()
        self.device = device
        self.resample = torchaudio.transforms.Resample(orig_freq=resample_samplerate, new_freq=output_samplerate).to(self.device)
        n_fft = (2*window_length_ms * resample_samplerate) // 2000
        hop_length = n_fft // 2
        self.invspec = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft,hop_length=hop_length).to(self.device)
       
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.to(self.device)

        # Convert to power spectrogram
        # Resample the output
        waveform = self.invspec(spec)
        resampled = self.resample(waveform)
        
        return resampled


# Complex 2d conv (code from: https://github.com/pheepa/DCUnet/tree/master)
class CConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output


# Complex transpose 2d conv (code from: https://github.com/pheepa/DCUnet/tree/master), modified
class CConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
        
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x, output_size):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real, output_size) - self.im_convt(x_im, output_size)
        ct_im = self.im_convt(x_real, output_size) + self.real_convt(x_im, output_size)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output


# Complex 2d batch norm (code from: https://github.com/pheepa/DCUnet/tree/master)
class CBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.stack([n_real, n_im], dim=-1)
        return output


# Encoder block (code from: https://github.com/pheepa/DCUnet/tree/master)
class Encoder(nn.Module):
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted


# Decoder block (code from: https://github.com/pheepa/DCUnet/tree/master), modified
class Decoder(nn.Module):
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x, output_size):
        
        conved = self.cconvt(x, output_size)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
            
        return output


#  Deep Complex U-Net (code from: https://github.com/pheepa/DCUnet/tree/master), modified
class DCUnet20(nn.Module):
    def __init__(self):
        super().__init__()
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(7,1), stride_size=(1,1), in_channels=1, out_channels=32, padding=(3,0))
        self.downsample1 = Encoder(filter_size=(1,7), stride_size=(1,1), in_channels=32, out_channels=32, padding=(0,3))
        self.downsample2 = Encoder(filter_size=(7,5), stride_size=(2,2), in_channels=32, out_channels=64, padding=(3,2))
        self.downsample3 = Encoder(filter_size=(7,5), stride_size=(2,1), in_channels=64, out_channels=64, padding=(3,2))
        self.downsample4 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=64, out_channels=64, padding=(2,1))
        self.downsample5 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=64, out_channels=64, padding=(2,1))
        self.downsample6 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=64, out_channels=64, padding=(2,1))
        self.downsample7 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=64, out_channels=64, padding=(2,1))
        self.downsample8 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=64, out_channels=64, padding=(2,1))
        self.downsample9 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=64, out_channels=90, padding=(2,1))
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=64, padding=(2,1))
        self.upsample1 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=128, out_channels=64, padding=(2,1))
        self.upsample2 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=128, out_channels=64, padding=(2,1))
        self.upsample3 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=128, out_channels=64, padding=(2,1))
        self.upsample4 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=128, out_channels=64, padding=(2,1))
        self.upsample5 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=128, out_channels=64, padding=(2,1))
        self.upsample6 = Decoder(filter_size=(7,5), stride_size=(2,1), in_channels=128, out_channels=64, padding=(3,2))
        self.upsample7 = Decoder(filter_size=(7,5), stride_size=(2,2), in_channels=128, out_channels=32, padding=(3,2))
        self.upsample8 = Decoder(filter_size=(1,7), stride_size=(1,1), in_channels=64, out_channels=32, padding=(0,3))
        self.upsample9 = Decoder(filter_size=(7,1), stride_size=(1,1), in_channels=64, output_padding=(0,1), padding=(3,0),
                                 out_channels=1, last_layer=True)
        
        
    def forward(self, x):
        x = torch.view_as_real(x.unsqueeze(1))
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)     
        d3 = self.downsample3(d2) 
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        d6 = self.downsample6(d5)
        d7 = self.downsample7(d6)
        d8 = self.downsample8(d7)
        d9 = self.downsample9(d8)
        
        # upsampling/decoding 
        u0 = self.upsample0(d9, output_size=d8[..., 0].size())
        # skip-connection
        c0 = torch.cat((u0, d8), dim=1)

        u1 = self.upsample1(c0, output_size=d7[..., 0].size())
        c1 = torch.cat((u1, d7), dim=1)

        u2 = self.upsample2(c1, output_size=d6[..., 0].size())
        c2 = torch.cat((u2, d6), dim=1)

        u3 = self.upsample3(c2, output_size=d5[..., 0].size())
        c3 = torch.cat((u3, d5), dim=1)

        u4 = self.upsample4(c3, output_size=d4[..., 0].size())
        c4 = torch.cat((u4, d4), dim=1)

        u5 = self.upsample5(c4, output_size=d3[..., 0].size())
        c5 = torch.cat((u5, d3), dim=1)

        u6 = self.upsample6(c5, output_size=d2[..., 0].size())
        c6 = torch.cat((u6, d2), dim=1)

        u7 = self.upsample7(c6, output_size=d1[..., 0].size())
        c7 = torch.cat((u7, d1), dim=1)
    
        u8 = self.upsample8(c7, output_size=d1[..., 0].size())
        c8 = torch.cat((u8, d0), dim=1)
        
        gains = self.upsample9(c8, output_size=x[..., 0].size())
        
        # u4 - the mask
        estimated_spec = gains * x
        
        return torch.view_as_complex(estimated_spec).squeeze(1), torch.view_as_complex(gains).squeeze(1)


def train(dataset, dataloader, model, preprocessor, loss_fn, optimizer, writer, save_path, epochs=1, device='cuda'):
    size = len(dataset)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 4])
    model.train()
    start_time = time.perf_counter()

    for epoch in range(epochs):
        for batch, (noisy_batch, clean_batch, _) in enumerate(dataloader):
            noisy_spec = preprocessor(noisy_batch).to(device)
            clean_spec = preprocessor(clean_batch).to(device)

            est_clean_spec, _ = model(noisy_spec)
            loss = loss_fn(noisy_spec, est_clean_spec, clean_spec)
            writer.add_scalar("Loss/Step", loss, epoch*size + batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (batch+1) % 20 == 0:
                torch.save(model, f"{save_path}/DCUnet20-checkpoint.pt")
                curr_time = time.perf_counter()
                batch_size = noisy_batch.shape[0]
                loss, current = loss.item(), 1 + (batch)*batch_size + epoch*size
                print(f"loss: {loss:>7f} [{current:>5d}/{size*epochs:>5d}] at {curr_time-start_time:>5f} sec")
                start_time = curr_time


# Loss function (code from: https://github.com/pheepa/DCUnet/tree/master), modified
class wsdr_fn(torch.nn.Module):
    def __init__(
            self,
            n_fft,
            hop_length,
            eps=1e-8,
            device='cuda'
        ):
            super().__init__()
            self.device = device
            self.invspec = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(self.device)
            self.eps = eps
    
    def forward(self, x_, y_pred_, y_true_: torch.Tensor) -> torch.Tensor:
        # to time-domain waveform
        y_true = self.invspec(y_true_)
        x = self.invspec(x_)
        y_pred = self.invspec(y_pred_)

        def sdr_fn(true, pred, eps=self.eps):
            num = torch.sum(true * pred, dim=1)
            den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
            return -(num / (den + eps))

        # true and estimated noise
        z_true = x - y_true
        z_pred = x - y_pred

        a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + self.eps)
        wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
        return torch.mean(wSDR)


def main():
    # mp.set_start_method('spawn')
    writer = SummaryWriter(log_dir="/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/tensorboard/DCU20")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    clean_zip = "/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/clean_trainset_28spk_wav.zip"
    noisy_zip = "/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/noisy_trainset_28spk_wav.zip"
    extracted_path = "/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/extracted"
    # Load data, create dataloaders, set parameters
    dataset = NoisySpeech(clean_zip, noisy_zip, extracted_path, device=device)

    _, _, input_samplerate = dataset.__getitem__(0)
    resample_samplerate = 16000
    window_length_ms = 30
    batch_size = 24
    n_fft = (2*window_length_ms * resample_samplerate) // 2000
    hop_length = n_fft // 2

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateNoisySpeech)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateNoisySpeech)

    # preprocessing stft not correct padding to max length atm
    enhancer = DCUnet20().to(device)
    save_path = "/home/aac/shared/teams/amd-internal/huggingface/datasets/megatron-data/sopiko/speech_data/models"
    loss_fn = wsdr_fn(n_fft, hop_length)
    epochs = 2
    optimizer = torch.optim.Adam(enhancer.parameters(), lr=0.001)
    preprocessor = PreProcessing(input_samplerate=input_samplerate, resample_samplerate=resample_samplerate, window_length_ms=window_length_ms, device=device)
    
    print("Starting Training")
    train(dataset, train_dataloader, enhancer, preprocessor, loss_fn, optimizer=optimizer, writer=writer, save_path=save_path,epochs=epochs, device=device)
    print("Finished Training")
    writer.flush()
    writer.close()
    torch.save(enhancer, "DCUnet20.pt")

def test(test_dataloader, enhancer, device):
    noisy_batch, clean_batch, sr = next(iter(test_dataloader))
    noisy_spec = preprocessor(noisy_batch).to(device)
    clean_spec = preprocessor(clean_batch).to(device)

    postprocessor = PostProcessing(output_samplerate=input_samplerate)

    enhancer.eval()
    with torch.no_grad():
        enhanced_spec, gains = enhancer(noisy_spec)

    enhanced_batch = postprocessor(enhanced_spec.to('cpu'))
    clean_audio = postprocessor(clean_spec.to('cpu'))
    noisy_audio = postprocessor(noisy_spec.to('cpu'))
    idx = np.random.randint(batch_size)
    print(idx)
    plt.figure(figsize=(8,3))
    plt.subplot(131)
    plt.imshow(noisy_spec[idx,:,:].to('cpu').abs().log().mT.numpy(),origin='lower', aspect="auto")
    plt.subplot(132)
    plt.imshow(enhanced_spec[idx,:,:].to('cpu').abs().log().mT.detach().numpy(),origin='lower', aspect="auto")
    plt.subplot(133)
    plt.imshow(clean_spec[idx,:,:].to('cpu').abs().log().mT.numpy(),origin='lower', aspect="auto")
    plt.show()

    import IPython
    IPython.display.display(IPython.display.Audio(noisy_batch[idx,:].detach().numpy(),rate=int(sr)))
    IPython.display.display(IPython.display.Audio(enhanced_batch[idx,:].detach().numpy(),rate=int(sr)))
    IPython.display.display(IPython.display.Audio(clean_batch[idx,:],rate=int(sr)))

if __name__ == "__main__":
    main()