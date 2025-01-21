import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import CollateFunctor, ScanDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import cosine_schedule_with_warmup, set_seed
from systematicity_entropy.models.cnn import CNN

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
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout value")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Grad clip value")
    parser.add_argument(
        "--acc",
        action="store_true",
        help="When activated, calculate accuracy per eval epoch",
    )
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
        "--kernel_size",
        type=int,
        default=5,
        help="The kernel size of the model convolution layers",
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
    return parser.parse_args()


def get_loss_from_cnn(batch: tuple, model: nn.Module) -> torch.Tensor:
    source_ids, source_mask, target_ids, target_mask = batch
    logits = model(source_ids, target_ids[:, :-1], source_mask.unsqueeze(1))
    loss = F.cross_entropy(
        logits.transpose(-2, -1), target_ids[:, 1:], ignore_index=model.pad_token
    )
    return loss


def get_accuracy_from_cnn(
    batch: tuple, idx_dict: dict, model: nn.Module
) -> int:
    max_new_tokens = MAX_GEN_TOKENS
    source_ids = batch[0].to(DEVICE)
    source_mask = batch[1].to(DEVICE)
    targets = batch[2].tolist()
    predictions = model.test_generate(
        source_ids,
        source_mask,
        max_new_tokens,
        idx_dict["pad_idx"],
        idx_dict["bos_idx"],
        idx_dict["eos_idx"],
        idx_dict["itoc"],
    )
    corrects = 0
    for pred, gold in zip(predictions, targets):
        print(pred, gold)
        eos_index = gold.index(idx_dict["eos_idx"])
        gold = gold[1:eos_index]  # skip BOS and EOS
        if pred == gold:
            corrects += 1
    return corrects


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
):
    prev_best_val_loss = 999.999
    patience = args.patience
    save_log: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }
    for epoch in range(args.epochs):
        logging.info(f"Starting training for epoch: {epoch}")
        pbar = tqdm(total=len(train_loader))
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = get_loss_from_cnn(batch, model)
            loss.backward()
            if args.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            pbar.update()
            pbar.set_postfix_str(f"Loss: {round(loss.item(), 3)}")
            train_loss += loss.item()

        logging.info("Evaluating....")
        correct = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                loss = get_loss_from_cnn(batch, model)
                val_loss += loss.item()
                if args.acc:
                    correct += get_accuracy_from_cnn(
                        batch, val_loader.dataset.idx_map, model
                    )

        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_loss = train_loss / len(train_loader)
        save_log["train_loss"].append(epoch_train_loss)
        save_log["val_loss"].append(epoch_val_loss)
        if args.acc:
            epoch_val_acc = correct / len(val_loader.dataset)
            wandb.log(
                {
                    "val_loss": epoch_val_loss,
                    "val_acc:": epoch_val_acc,
                    "train_loss": epoch_train_loss,
                }
            )
            logging.info(
                f"Train loss: {epoch_train_loss}, val loss: {epoch_val_loss}, val accuracy {epoch_val_acc}"
            )
            save_log["val_acc"].append(epoch_val_acc)
        else:
            wandb.log({"val_loss": epoch_val_loss, "train_loss": epoch_train_loss})
            logging.info(f"Train loss: {epoch_train_loss}, val loss: {epoch_val_loss}")
        pbar.close()

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
        "type": "CNN"
    }
    wandb.init(project=args.wandb, config=wandb_config)

    training_set = DATASET_MAP[args.task](
        args.train_path, device=DEVICE, task=args.task
    )
    validation_set = DATASET_MAP[args.task](
        args.val_path, device=DEVICE, task=args.task
    )
    collator = CollateFunctor(training_set.pad_idx)
    train_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        validation_set, batch_size=16, collate_fn=collator, shuffle=False
    )
    vocab_size = len(training_set.ctoi.keys())

    model = CNN(
        args.hidden_size,
        args.kernel_size,
        args.dropout,
        vocab_size,
        vocab_size,
        args.layers,
        args.layers,
        training_set.pad_idx,
    ).to(DEVICE)

    print(model)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of parameters: {params}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98)
    )
    max_steps = len(train_loader) * args.epochs
    scheduler = cosine_schedule_with_warmup(
        optimizer, int(max_steps * 0.06), max_steps, 0.1
    )

    train(model, train_loader, val_loader, optimizer, scheduler, args)


if __name__ == "__main__":
    main()
