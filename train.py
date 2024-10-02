import torch
import os
import data
import argparse
import logging

from losses import MelLoss, CombinedSpectralLoss
from model import VADNoiseModelEnhancer, BLSTM, CRN
from data import NoisySpeech, collate_fn
from utils import PreProcessor, PostProcessor, train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True, type=str, help="The name of the run")

    parser.add_argument("--model", default="crn", type=str)
    parser.add_argument("--loss", default="SpecLoss", type=str)

    parser.add_argument("--n_fft", default=200, type=int)
    parser.add_argument("--n_mels", default=20, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-3, type=int)
    parser.add_argument("--weight_decay", default=0, type=int)

    parser.add_argument("--warmup_steps", default=30, type=int)
    parser.add_argument("--save_steps", default=200, type=int)
    
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--out_dir", default="./results", type=str)
    parser.add_argument("--log_dir", default="./results/log.txt", type=str)

    parser.add_argument('--device', default="cpu", type=str)

    return parser.parse_args()


def init_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler = logging.FileHandler(filename=log_dir, mode="w")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    args = parse_args()

    run_root_dir = args.out_dir + "/" + args.run_name
    save_dir = run_root_dir + "/model_checkpoints"
    log_dir = run_root_dir + "/log.txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = init_logger(log_dir)
    logger.info(f"Run name: {args.run_name}")

    print("Preparing dataset...")
    dataset = NoisySpeech(args.data_dir, device=args.device)

    _, _, input_samplerate = dataset.__getitem__(0)
    resample_samplerate = 16000

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    preprocessor = PreProcessor(input_samplerate=input_samplerate, resample_samplerate=resample_samplerate, n_fft=args.n_fft)
    postprocessor = PostProcessor(output_samplerate=input_samplerate, resample_samplerate=resample_samplerate, n_fft=args.n_fft)

    print(f"Loading model {args.model}...")
    if args.model == "baseline":
        model = VADNoiseModelEnhancer(
            spec_size=args.n_fft // 2 + 1,
            input_samplerate=input_samplerate,
            enhancer_size=256,
            noise_model_size=32,
            vad_model_size=128,
            n_mels=args.n_mels,
            smoothing=0.8,
            device=args.device,
        )
    elif args.model == "lstm":
        model = BLSTM(
            spec_size=args.n_fft // 2 + 1,
            lstm_hidden_size=256,
            lstm_layers=3,
            attention_heads=3,
            n_mels=args.n_mels,
            dropout=0.4,
            device=args.device,
        )
    elif args.model == "crn":
        model = CRN(
            n_fft=args.n_fft,
            gru_layers=1,
            device=args.device,
        )
    model.to(args.device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, threshold=0.08, min_lr=5e-5)

    if args.loss == "MelLoss":
        criterion = MelLoss(sample_rate=resample_samplerate, n_stft=args.n_fft // 2 + 1, n_mels=32, device=args.device)
    elif args.loss == "SpecLoss":
        criterion = CombinedSpectralLoss(alpha=0.8)

    print("Starting training...")
    train(
        dataloader=train_dataloader,
        dataset=train_dataset,
        model=model,
        preprocessor=preprocessor,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        save_dir=save_dir,
        save_steps=args.save_steps,
        logger=logger,
    )

