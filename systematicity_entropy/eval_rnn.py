import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from dataset import ScanDataset
from tqdm import tqdm
from utils import set_seed

import wandb
from systematicity_entropy.models.rnn import BasicSeq2Seq

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
    parser.add_argument("--model_path", type=Path, help="Checkpoint to load")
    parser.add_argument(
        "--wandb", type=str, default="debug_eval", help="Wandb project name"
    )
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
        "--type",
        type=str,
        default="GRU",
        choices=["GRU", "LSTM", "RNN", "GRNN", "GGRU", "GLSTM"],
        help="Which RNN type to use",
    )
    return parser.parse_args()


@torch.no_grad
def test_scan_gen(model: nn.Module, val_data: ScanDataset) -> float:
    """
    Tests the policy on the validation dataset

    Args:
        model (nn.Module): The model
        val_data (Dataset): The validation/test dataset object

    Returns:
        float: the test accuracy
    """
    correct = 0
    errors = 0
    pbar = tqdm(total=len(val_data))
    for i, sample in enumerate(val_data):
        pbar.update()
        input_ids, target_ids = sample
        target_length = target_ids.size(0)
        logits = model(input_ids, target_length)
        _, sentence_ids = logits.data.topk(1)
        try:
            eos_location = (sentence_ids == val_data.eos_idx).nonzero()[0][0]
        except:
            eos_location = len(sentence_ids) - 2
        model_sentence = sentence_ids[: eos_location + 1].squeeze()
        target_eos_location = (target_ids == val_data.eos_idx).nonzero()[0][0]
        target_sentence = target_ids[: target_eos_location + 1]
        if len(model_sentence) != len(target_sentence):
            errors += 1
        else:
            if torch.equal(model_sentence, target_sentence):
                correct += 1

    pbar.close()
    return correct / len(val_data)


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

    model = BasicSeq2Seq(
        vocab_size=vocab_size,
        encoder_hidden_size=args.hidden_size,
        decoder_hidden_size=args.hidden_size,
        layer_type=args.type,
        use_attention=True,
        drop_rate=0.0,
        bidirectional=True,
        num_layers=args.layers,
        bos_token_idx=val.bos_idx,
        eos_token_idx=val.eos_idx,
    )
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.encoder.eval()
    model.decoder.eval()
    config = {
        "val_data": str(args.val_path),
        "model_path": str(args.model_path),
        "seed": args.seed,
        "entropy_level": args.entropy_level,
        "size": args.size,
        "type": args.type,
    }
    wandb_name = f"{str(args.model_path)}"
    wandb.init(name=wandb_name, project=args.wandb, config=config)

    logging.info("Calculating final accuracy")
    final_acc = test_scan_gen(model, val)
    wandb.log({"final_acc": final_acc})
    logging.info(f"Final accuracy: {final_acc}")


if __name__ == "__main__":
    main()
