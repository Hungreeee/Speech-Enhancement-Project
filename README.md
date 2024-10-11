# Speech-Enhancement-Project

**Team members:** Diana Bacircea, Jussi H, Ville Huhtala, Sopiko Kurdadze, Hung Nguyen

A modified version of [Zhang W. et. al, Magnitude-and-phase-aware Speech Enhancement with Parallel Sequence Modeling](https://arxiv.org/abs/2310.07316).

The model has the CRN architecture, sort of a UNet with GRU layers as the bottleneck. The idea is that UNet can downsample/upsample the frequency-domain features using a set of convolutional windows, keeping the time-domain features (almost) intact. The bottleneck learns the downsampled features using a unidirectional GRU layer to preserve causality of the time-domain axis. The model also utilizes attention layers to attend the magnitude to the phase features, ensuring that the phase information, which is often very difficult to model correctly, can be less noisy where there is no speech. 

Experiments with different model can be found in the directory `experiments/`, please refer to it for more details. 
