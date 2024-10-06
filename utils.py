import time
import torch
from torchaudio import transforms
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

def calc_oracle_gains(noisy_spec: torch.Tensor, clean_spec: torch.Tensor):
    return (clean_spec.abs()/noisy_spec.abs().clamp(min=1e-6)).clamp(max=1)


class PreProcessor(torch.nn.Module):
    def __init__(
        self,
        input_samplerate = 16000,
        resample_samplerate = 16000,
        n_fft = 480,
        power=None
    ):
        super().__init__()
        self.output_size = (n_fft+2)//2
        self.resample = transforms.Resample(input_samplerate, resample_samplerate)
        self.transform = transforms.Spectrogram(n_fft=n_fft, power=power, window_fn=torch.hann_window, normalized=True, center=True)

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
        self.transform = transforms.InverseSpectrogram(n_fft=n_fft, window_fn=torch.hann_window, normalized=True, center=True)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.permute(0, 2, 1)
        waveform = self.transform(spec)
        resampled = self.resample(waveform)
        return resampled


def print_and_log(logger, variable):
    logger.info(variable)
    print(variable)


def train(
    dataloader, 
    dataset, 
    model, 
    preprocessor,
    loss_fn, 
    optimizer, 
    epochs,
    scheduler,
    save_steps,
    save_dir,
    logger,
):
    size = len(dataset)
    model.train()
    start_time = time.perf_counter()

    for epoch in range(epochs):
        loss_total = 0.0
        for step, (noisy_batch, clean_batch, _) in enumerate(dataloader):
            batch_size = noisy_batch.shape[0]

            noisy_spec = preprocessor(noisy_batch)
            clean_spec = preprocessor(clean_batch)

            est_clean_spec = model(noisy_spec)
            loss = loss_fn(est_clean_spec, clean_spec)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step(loss)
    
            loss = loss.item()
            loss_total += loss
            curent_step = 1 + (step) * batch_size + epoch * size

            if (step + 1) % 10 == 0:
                curr_time = time.perf_counter()
                print_and_log(logger, f"loss: {loss:>7f}  [{curent_step:>5d}/{size*epochs:>5d}] at {curr_time-start_time:>5f} sec")
                start_time = curr_time
            
            if (step + 1) % 5 == 0:
                print_and_log(logger, f"Model checkpoints saved to {save_dir}")
                torch.save(model, save_dir + f"/model_{curent_step}.pt")

        print_and_log(logger, f"EPOCH SUMMARY - loss: {loss_total / len(dataloader):>7f}  [{epoch:>5d}/{epochs:>5d}]")
        start_time = curr_time
        
    torch.save(model, save_dir + f"/model_final.pt")


def eval(
    dataloader, 
    dataset, 
    model, 
    preprocessor,
    postprocessor, 
    loss_fn, 
    optimizer, 
    epochs,
    scheduler,
    save_steps,
    save_dir,
    logger
):
    size = len(dataset)
    model.eval()
    start_time = time.perf_counter()

    for epoch in range(epochs):
        loss_total = 0.0
        for step, (noisy_batch, clean_batch, _) in enumerate(dataloader):
            batch_size = noisy_batch.shape[0]

            noisy_spec = preprocessor(noisy_batch)
            clean_spec = preprocessor(clean_batch)

            est_clean_spec = model(noisy_spec)
            loss = loss_fn(est_clean_spec, clean_spec)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step(loss)
    
            loss = loss.item()
            loss_total += loss

            est_clean_batch = postprocessor(est_clean_spec)

            pesq = perceptual_evaluation_speech_quality(est_clean_batch, clean_batch, fs=16000, mode='wb')

            curent_step = 1 + (step) * batch_size + epoch * size

            if (step + 1) % 10 == 0:
                curr_time = time.perf_counter()
                print_and_log(logger, f"loss: {loss:>7f}  [{curent_step:>5d}/{size*epochs:>5d}] at {curr_time-start_time:>5f} sec")
                print_and_log(logger, f"PESQ Score: {pesq}")
                start_time = curr_time
            
            if (step + 1) % 5 == 0:
                print_and_log(logger, f"Model checkpoints saved to {save_dir}")
                torch.save(model, save_dir + f"/model_{curent_step}.pt")

        print_and_log(logger, f"EPOCH SUMMARY - loss: {loss_total / len(dataloader):>7f}  [{epoch:>5d}/{epochs:>5d}]")
        start_time = curr_time
        