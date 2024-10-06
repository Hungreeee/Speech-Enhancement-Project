import zipfile
import time
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import os
import scipy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# More efficient data loading


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


# Move processing on device


class PreProcessing(torch.nn.Module):
    def __init__(
        self,
        input_samplerate=16000,
        resample_samplerate=16000,
        window_length_ms=30,
        device='cuda'  # Add device parameter
    ):
        super().__init__()
        self.device = device
        self.resample = torchaudio.transforms.Resample(orig_freq=input_samplerate, new_freq=resample_samplerate).to(self.device)
        n_fft = (2 * window_length_ms * resample_samplerate) // 2000
        hop_length = n_fft // 2
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None, hop_length=hop_length).to(self.device)
        self.output_size = (n_fft + 2) // 2

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Ensure waveform is on the correct device
        waveform = waveform.to(self.device)
        
        # Resample the input
        resampled = self.resample(waveform)
        
        # Convert to power spectrogram
        spec = self.spec(resampled)
        
        return spec


class PostProcessing(torch.nn.Module):
    def __init__(
        self,
        output_samplerate=16000,
        resample_samplerate=16000,
        window_length_ms=30,
        device='cuda'  # Add device parameter
    ):
        super().__init__()
        self.device = device
        self.resample = torchaudio.transforms.Resample(orig_freq=resample_samplerate, new_freq=output_samplerate).to(self.device)
        n_fft = (2 * window_length_ms * resample_samplerate) // 2000
        hop_length = n_fft // 2
        self.invspec = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(self.device)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # Ensure spec is on the correct device
        spec = spec.to(self.device)
        
        # Convert to waveform from spectrogram
        waveform = self.invspec(spec)
        
        # Resample the output
        resampled = self.resample(waveform)
        
        return resampled
    