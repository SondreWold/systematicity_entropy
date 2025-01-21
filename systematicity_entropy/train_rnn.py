import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dataset import ScanDataset
from tqdm import tqdm
from utils import set_seed

from systematicity_entropy.models.rnn import BasicSeq2Seq
from wandb import wandb  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_TOKENS = 64
CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
MODEL_FOLDER = CURR_FILE_DIR / "trained_models"
DATASET_MAP = {
        "scan": ScanDataset,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description="Training script for systematicity experiments"
            )
    parser.add_argument("--train_path", type=Path, help="Train data path")
    parser.add_argument("--val_path", type=Path, help="Val data path")
    parser.add_argument("--save_path", type=Path, help="Where to save the model")
    parser.add_argument(
            "--wandb", type=str, default="ood_debug", help="Wandb project name"
            )
    parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Learning rate for the optimizer.",
            )
    parser.add_argument(
            "--seed", type=int, default=42, help="Random seed for reproducibility."
            )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout value")
    parser.add_argument(
            "--forcing", action="store_true", help="Whether or not to use teacher forcing"
            )
    parser.add_argument(
            "--grad_acc",
            action="store_true",
            help="Whether or not to use gradient accumulation",
            )
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Grad clip value")
    parser.add_argument(
            "--epochs", type=int, default=10, help="Number of training epochs"
            )
    parser.add_argument(
            "--batch_size", type=int, default=512, help="Default training batch size"
            )
    parser.add_argument(
            "--layers", type=int, default=6, help="Number of Transformer decoder layers"
            )
    parser.add_argument(
            "--hidden_size",
            type=int,
            default=384,
            help="The hidden size of the model layers",
            )
    parser.add_argument(
            "--patience",
            type=int,
            default=3,
            help="How many epochs to allow val loss increase",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="scan",
            choices=["scan"],
            help="Which task to train the model on",
            )
    parser.add_argument(
            "--type",
            type=str,
            default="GRU",
            choices=["GRU", "LSTM", "RNN", "GRNN", "GGRU", "GLSTM"],
            help="Which RNN type to use",
            )
    return parser.parse_args()


def train(
        model: torch.nn.Module,
        train_dataset: ScanDataset,
        val_dataset: ScanDataset,
        e_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        args: argparse.Namespace,
        ):
    prev_best_val_loss = 999.999
    patience = args.patience
    criterion = nn.NLLLoss()
    save_log: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            }
    for epoch in range(args.epochs):
        logging.info(f"Starting training for epoch: {epoch}")
        model.encoder.train()
        model.decoder.train()
        train_loss = 0.0
        val_loss = 0.0
        pbar = tqdm(total=len(train_dataset))
        for idx, sample in enumerate(train_dataset):
            pbar.update()
            input_ids, target_ids = sample
            input_ids = input_ids.view(-1, 1)
            target_ids = target_ids.view(-1, 1)
            target_length = target_ids.size(0)
            if args.forcing:
                if np.random.uniform(0, 1) > 0.5:
                    teacher_forcing = True
                else:
                    teacher_forcing = False
            else:
                teacher_forcing = False
            logits = model(
                    input_ids,
                    target_tensor=target_ids,
                    use_teacher_forcing=teacher_forcing,
                    target_length=target_length,
                    )
            loss = 0.0

            for di in range(target_length):
                decoder_output = logits[di]
                loss += criterion(decoder_output[None, :], target_ids[di])
                _, decoder_output_symbol = decoder_output.topk(1)
                if decoder_output_symbol.item() == train_dataset.eos_idx:
                    break

            loss = loss / target_length
            loss.backward()
            """
            torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(), args.grad_clip
                    )
            torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(), args.grad_clip
                    )
            """
            if args.grad_acc:
                if ((idx + 1) % args.batch_size == 0) or (
                        idx + 1 == len(train_dataset)
                        ):
                    e_optimizer.step()
                    d_optimizer.step()
                    e_optimizer.zero_grad()
                    d_optimizer.zero_grad()
            else:
                e_optimizer.step()
                d_optimizer.step()
                e_optimizer.zero_grad()
                d_optimizer.zero_grad()

            train_loss += loss.item()

        pbar.close()
        model.encoder.eval()
        model.decoder.eval()
        pbar = tqdm(total=len(val_dataset))
        for i, sample in enumerate(val_dataset):
            pbar.update()
            input_ids, target_ids = sample
            input_ids = input_ids.view(-1, 1)
            target_ids = target_ids.view(-1, 1)
            target_length = target_ids.size(0)
            logits = model(input_ids, target_length=target_length)
            loss = 0.0

            for di in range(target_length):
                decoder_output = logits[di]
                loss += criterion(decoder_output[None, :], target_ids[di])
                _, decoder_output_symbol = decoder_output.topk(1)
                if decoder_output_symbol.item() == val_dataset.eos_idx:
                    break
            loss = loss / target_length
            val_loss += loss.item()
        pbar.close()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        wandb.log({"val_loss": epoch_val_loss, "train_loss": epoch_train_loss})
        logging.info(f"Train loss: {epoch_train_loss}, val loss: {epoch_val_loss}")
        save_log["train_loss"].append(epoch_train_loss)
        save_log["val_loss"].append(epoch_val_loss)

        if epoch_val_loss < prev_best_val_loss:
            logging.info("Saving checkpoint...")
            patience = args.patience
            prev_best_val_loss = epoch_val_loss
            save_folder = MODEL_FOLDER / args.save_path
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_folder / "final_model.pt")
            with open(save_folder / "log.json", "w") as f:
                json.dump(save_log, f)
        else:
            patience -= 1

        if patience == 0:
            logging.info("Patience ran out... breaking training loop")
            break

        wandb.log({"best_val_loss": prev_best_val_loss})


def main():
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            )
    args = parse_args()
    set_seed(args.seed)
    wandb_config = {
            "lr": args.learning_rate,
            "val_data": str(args.val_path),
            "train_data": str(args.train_path),
            "save_path": str(args.save_path),
            "epochs": args.epochs,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "grad_clip": args.grad_clip,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "rnn_type": args.type,
            "teacher_forcing": args.forcing,
            }
    wandb.init(project=args.wandb, config=wandb_config)

    training_set = DATASET_MAP[args.task](
            args.train_path, device=DEVICE, task=args.task
            )
    validation_set = DATASET_MAP[args.task](
            args.val_path, device=DEVICE, task=args.task
            )
    vocab_size = len(training_set.ctoi.keys())
    model = BasicSeq2Seq(
            vocab_size=vocab_size,
            encoder_hidden_size=args.hidden_size,
            decoder_hidden_size=args.hidden_size,
            layer_type=args.type,
            use_attention=True,
            drop_rate=args.dropout,
            bidirectional=True,
            num_layers=args.layers,
            bos_token_idx=training_set.bos_idx,
            eos_token_idx=training_set.eos_idx,
            )
    model.encoder.to(DEVICE)
    model.decoder.to(DEVICE)
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Data device: {training_set[0][0].device}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of parameters: {params}")

    e_optimizer = torch.optim.AdamW(
            model.encoder.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            )
    d_optimizer = torch.optim.AdamW(
            model.decoder.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            )
    train(model, training_set, validation_set, e_optimizer, d_optimizer, args)


if __name__ == "__main__":
    main()
