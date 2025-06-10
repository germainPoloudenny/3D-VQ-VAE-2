import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pathlib import Path

from vqvae.model import VQVAE
from utils import AmplitudeDataModule


def parse_arguments():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)

    parser.add_argument("--dataset_path", type=Path, default="/home/gpoloudenny/Projects/crystallography/data/mp20/amplitudes/normalized.h5")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-key", type=str, default="amplitudes")
    parser.add_argument("--hkl-max-index", type=int, default=10)

    parser.set_defaults(
        gpus="-1",
        accelerator="ddp",
        benchmark=True,
        num_sanity_val_steps=0,
        precision=16,
        log_every_n_steps=50,
        val_check_interval=0.5,
        flush_logs_every_n_steps=100,
        weights_summary="full",
        max_epochs=int(1e5),
    )

    args = parser.parse_args()
    args.dataset_path = str(args.dataset_path.resolve())
    return args


def main(args):
    torch.cuda.empty_cache()
    pl.trainer.seed_everything(seed=42)

    datamodule = AmplitudeDataModule(
        path=args.dataset_path,
        batch_size=args.batch_size,
        data_key=args.data_key,
        hkl_max_index=args.hkl_max_index,
        num_workers=5,
    )

    model = VQVAE(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1, save_last=True, monitor="val_recon_loss_mean"
    )

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(parse_arguments())