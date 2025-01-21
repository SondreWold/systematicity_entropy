import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from dataset import CollateFunctor, ScanDataset
from systematicity_entropy.models.cnn import CNN
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import set_seed

import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_MAP = {
    "scan": ScanDataset,
    "pcfg": ScanDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a machine learning model."
    )
    parser.add_argument("--val_path", type=Path, help="Val data path")
    parser.add_argument(
        "--entropy_level", type=str, default="not_set", help="Entropy level"
    )
    parser.add_argument("--size", type=str, default="not_set", help="Val data path")
    parser.add_argument(
        "--wandb", type=str, default="ood_generic", help="Wandb project name"
    )
    parser.add_argument("--model_path", type=Path, help="Checkpoint to load")
    parser.add_argument(
        "--layers", type=int, default=6, help="Number of Transformer decoder layers"
    )
    parser.add_argument(
        "--heads", type=int, default=6, help="Number of attention heads"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=384,
        help="The hidden size of the model layers",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="number",
        choices=["number", "scan", "pcfg"],
        help="Which task to train the model on",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=7,
        help="The kernel size of the model convolution layers",
    )
    return parser.parse_args()


@torch.no_grad
def test_scan_gen(model: nn.Module, val_data: Dataset) -> float:
    """
    Tests the policy on the validation dataset

    Args:
        model (nn.Module): The policy
        val_data (Dataset): The validation/test dataset object

    Returns:
        float: the test accuracy
    """
    max_new_tokens = 64
    correct = 0
    errors = 0
    collator = CollateFunctor(val_data.pad_idx)
    test_loader = DataLoader(val_data, batch_size=8, collate_fn=collator, shuffle=False)
    for batch in tqdm(test_loader):
        source_ids = batch[0].to(DEVICE)
        source_mask = batch[1].to(DEVICE)
        targets = batch[2].tolist()
        #print(source_mask.shape)
        #print(source_ids.shape)
        predictions = model.test_generate(
            source_ids,
            source_mask,
            max_new_tokens,
            val_data.pad_idx,
            val_data.bos_idx,
            val_data.eos_idx,
            val_data.itoc,
        )
        for pred, gold in zip(predictions, targets):
            eos_index = gold.index(val_data.eos_idx)
            gold = gold[1:eos_index]  # skip BOS and EOS
            if pred == gold:
                correct += 1
            else:
                errors += 1

    return correct / (correct + errors)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    set_seed(args.seed)

    val = DATASET_MAP[args.task](args.val_path, device=DEVICE, task=args.task)
    vocab_size = len(val.ctoi.keys())
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    model = CNN(
        args.hidden_size,
        args.kernel_size,
        0,
        vocab_size,
        vocab_size,
        args.layers,
        args.layers,
        val.pad_idx,
    )


    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    config = {
        "val_data": str(args.val_path),
        "model_path": str(args.model_path),
        "seed": args.seed,
        "entropy_level": args.entropy_level,
        "size": args.size,
    }
    wandb_name = f"{str(args.model_path)}"
    wandb.init(name=wandb_name, project=args.wandb, config=config)

    logging.info("Calculating final accuracy")
    final_acc = test_scan_gen(model, val)
    wandb.log({"final_acc": final_acc})
    logging.info(final_acc)


if __name__ == "__main__":
    main()
