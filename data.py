import zipfile
import torch
from torch.utils.data import Dataset
import scipy


class NoisySpeech(Dataset):
    def __init__(self, path, device='cpu'):
        self.path = path
        self.device = device
        self.__clean_zip__ = 'clean_trainset_28spk_wav.zip'
        self.__noisy_zip__ = 'noisy_trainset_28spk_wav.zip'
        with zipfile.ZipFile(self.path + "/" + self.__clean_zip__, 'r') as clean_zip:
            cleanlist = clean_zip.namelist()
        self.__clean_wav_list__ = [s for s in cleanlist if s[-4:] == '.wav']
        with zipfile.ZipFile(self.path + "/" + self.__noisy_zip__, 'r') as noisy_zip:
            noisylist = noisy_zip.namelist()
        self.__noisy_wav_list__ = [s for s in noisylist if s[-4:] == '.wav']

    def __len__(self):
        return len(self.__noisy_wav_list__)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.path + "/" + self.__clean_zip__, 'r') as clean_zip:
            with clean_zip.open(self.__clean_wav_list__[idx]) as clean_wav_file:                
                sr, np_clean_audio = scipy.io.wavfile.read(clean_wav_file)            
        with zipfile.ZipFile(self.path + "/" + self.__noisy_zip__, 'r') as noisy_zip:
            with noisy_zip.open(self.__noisy_wav_list__[idx]) as noisy_wav_file:                
                sr, np_noisy_audio = scipy.io.wavfile.read(noisy_wav_file)            
        return torch.tensor(np_noisy_audio), torch.tensor(np_clean_audio), torch.tensor(sr)

def collate_fn(itemlist):
    buffer_len = 6*48000 # Maximum length is 60 sec at 48kHz
    sample_len = buffer_len
    noisy_batch, clean_batch = torch.Tensor(0), torch.Tensor(0)
    
    for noisy, clean, sr in itemlist:
        sample_len = min(len(noisy),sample_len)
        noisy_padded, clean_padded = torch.zeros(buffer_len), torch.zeros(buffer_len)
        noisy_padded[0:sample_len], clean_padded[0:sample_len] = noisy[0:sample_len], clean[0:sample_len]
        noisy_batch = torch.cat((noisy_batch, noisy_padded.unsqueeze(0)))
        clean_batch = torch.cat((clean_batch, clean_padded.unsqueeze(0)))
        
    return noisy_batch[:,0:sample_len], clean_batch[:,0:sample_len], sr
    