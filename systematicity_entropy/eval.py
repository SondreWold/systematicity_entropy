import argparse
import json
import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from dataset import CollateFunctor, ScanDataset, SequenceDataset
from model import GPT, GPTConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from transformer import Transformer
from transformer_rope import Transformer
from utils import set_seed

TERM_EXTRACTION_PATTERN = r"(\d+)\-(\d+)=(\d+)"
TERM_EXTRACTION_PATTERN_2 = r"(\d+)=(\d+)"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_MAP = {
    "number": SequenceDataset,
    "scan": ScanDataset,
    "pcfg": ScanDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a machine learning model."
    )
    parser.add_argument(
        "--block_size", type=int, default=32, help="The block size of the transformer"
    )
    parser.add_argument("--val_path", type=Path, help="Val data path")
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
    max_new_tokens = 256
    correct = 0
    errors = 0
    collator = CollateFunctor(val_data.pad_idx)
    test_loader = DataLoader(val_data, batch_size=8, collate_fn=collator, shuffle=False)
    for batch in tqdm(test_loader):
        source_ids = batch[0].to(DEVICE)
        source_mask = batch[1].to(DEVICE)
        targets = batch[2].tolist()
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


@torch.no_grad
def test_number_gen(
    model: nn.Module, val_data: Dataset
) -> tuple[float, list[dict], list[dict]]:
    """
    Tests the policy on the validation dataset

    Args:
        model (nn.Module): The policy
        val_data (Dataset): The validation/test dataset object

    Returns:
        float: the test accuracy
    """
    max_new_tokens = 9  # 8 new seq. elements + EOS
    temperature = 1.0  # Deterministic scaling
    top_k = 1

    x_tokenized = []
    gold = []
    validation_samples = val_data.raw_data.split("\n")
    for sample in validation_samples:
        matched_terms = re.match(TERM_EXTRACTION_PATTERN_2, sample)
        if matched_terms:
            term1, result = matched_terms.groups()
            tokenized_sample = val_data.encode(f"{val_data.bos_token}{term1}=")
            gold.append(result)
            x_tokenized.append(tokenized_sample)

    n_corr = 0.0
    correct_objs = []
    error_objs = []
    with torch.no_grad():
        for sample, correct in tqdm(zip(x_tokenized, gold)):
            sample = torch.tensor([sample]).to(DEVICE)
            predictions = model.test_generate(
                sample, max_new_tokens, temperature=temperature, top_k=top_k
            ).squeeze(0)
            decoded = val_data.decode(predictions).strip()
            decoded = decoded[1:]  # Skip SOS token
            matched_terms_dec = re.match(TERM_EXTRACTION_PATTERN_2, decoded)
            if matched_terms_dec:
                term1, result = matched_terms_dec.groups()
                obj = {
                    "term1": term1,
                    # "term2": term2,
                    "prediction": result,
                    "gold": correct,
                }
                if correct == result:
                    n_corr += 1.0
                    correct_objs.append(obj)
                else:
                    error_objs.append(obj)
    acc = n_corr / len(gold)
    return acc, correct_objs, error_objs


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    set_seed(args.seed)
    val = DATASET_MAP[args.task](
        args.val_path, args.block_size, device=DEVICE, task=args.task
    )
    vocab_size = len(val.ctoi.keys())
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    if args.task == "number":
        model_config = GPTConfig(
            block_size=args.block_size,
            n_embd=args.hidden_size,
            n_head=args.heads,
            n_layer=args.layers,
            vocab_size=vocab_size,
        )
        model = GPT(model_config)
    else:
        model = Transformer(
            vocab_size, vocab_size, args.layers, args.hidden_size, 0.1, val.pad_idx
        )
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    config = {
        "val_data": str(args.val_path),
        "model_path": str(args.model_path),
        "seed": args.seed,
    }
    wandb.init(project="ood_generic", config=config)

    logging.info("Calculating final accuracy")
    if args.task == "number":
        final_acc, cs, es = test_number_gen(model, val)
        folder = args.model_path.parents[0]
        with open(folder / f"{str(args.val_path.stem)}_corrects.json", "w") as f:
            json.dump(cs, f)
        with open(folder / f"{str(args.val_path.stem)}_errors.json", "w") as f:
            json.dump(es, f)
    else:
        final_acc = test_scan_gen(model, val)
    wandb.log({"final_acc": final_acc})
    logging.info(final_acc)


if __name__ == "__main__":
    main()
