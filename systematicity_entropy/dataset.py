import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"
TRANSLATION_TOKEN = "[TRANS]"
CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
SCAN_TOKENIZER_PATH = CURR_FILE_DIR / "../static/scan_tokenizer.json"


class ScanDataset(Dataset):
    def __init__(
        self,
        path: str,
        device: str = "cpu",
        task: str = "scan",
        recreate_tokenizer: bool = False,
    ):
        self.pairs = self.read_pairs(Path(path))
        self.task = task
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.translation_token = TRANSLATION_TOKEN
        tokenizer_path = SCAN_TOKENIZER_PATH
        if recreate_tokenizer:
            self.ctoi, self.itoc = self.create_tokenizer(tokenizer_path)
        else:
            self.ctoi, self.itoc = self.load_tokenizer(tokenizer_path)

        self.pad_idx = self.ctoi[PAD_TOKEN]
        self.bos_idx = self.ctoi[BOS_TOKEN]
        self.eos_idx = self.ctoi[EOS_TOKEN]
        self.idx_map = {
            "pad_idx": self.pad_idx,
            "bos_idx": self.bos_idx,
            "eos_idx": self.eos_idx,
            "itoc": self.itoc,
            "ctoi": self.ctoi,
        }
        self.device = device
        self.inputs = []
        self.outputs = []
        for input_str, output_str in self.pairs:
            tokenized_input = self.encode(f"{input_str} {EOS_TOKEN}")
            self.inputs.append(tokenized_input)
            tokenized_output = self.encode(f"{BOS_TOKEN} {output_str} {EOS_TOKEN}")
            self.outputs.append(tokenized_output)

    def create_tokenizer(self, save_path: Path) -> tuple[dict, dict]:
        vocab = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, TRANSLATION_TOKEN]
        for input_str, output_str in self.pairs:
            input_tokens = input_str.split(" ")
            output_tokens = output_str.split(" ")
            vocab += input_tokens
            vocab += output_tokens
        vocab = list(set(vocab))
        ctoi = {}
        for i, vocab_item in enumerate(vocab):
            ctoi[vocab_item] = i
        itoc = {i: char for char, i in ctoi.items()}

        with open(save_path, "w") as f:
            json.dump(ctoi, f)
        return ctoi, itoc

    def load_tokenizer(self, tokenizer_path):
        with open(tokenizer_path, "r") as f:
            ctoi = json.load(f)
        itoc = {i: char for char, i in ctoi.items()}
        return ctoi, itoc

    def read_pairs(self, path: Path):
        try:
            lines = open(path, encoding="utf-8").read().strip().split("\n")
        except IOError as e:
            raise Exception("Failed to open SCAN data file: ", e)

        pairs = []
        for line in lines:
            inl, outl = line.split("OUT:")
            inl = inl.replace("IN:", "").strip()
            outl = outl.strip()
            pairs.append([inl, outl])
        return pairs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.tensor(self.inputs[idx]).to(self.device)
        targets = torch.tensor(self.outputs[idx]).to(self.device)
        return inputs, targets

    def decode(self, tokens: list[torch.Tensor]) -> str:
        return " ".join([self.itoc[c.item()] for c in tokens])

    def encode(self, sequence: str) -> list:
        return [self.ctoi[c] for c in sequence.split(" ")]


class CollateFunctor:
    def __init__(self, pad_idx: int) -> None:
        self.pad_id = pad_idx

    def __call__(
        self, sentences: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_ids, target_ids = zip(*sentences)
        source_ids, source_mask = self.collate_sentences(source_ids)
        target_ids, target_mask = self.collate_sentences(target_ids)
        return source_ids, source_mask, target_ids, target_mask

    def collate_sentences(self, sentences: list) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = [sentence.size(0) for sentence in sentences]
        max_length = max(lengths)
        subword_ids = torch.stack(
            [
                F.pad(sentence, (0, max_length - length), value=self.pad_id)
                for length, sentence in zip(lengths, sentences)
            ]
        )
        attention_mask = subword_ids == self.pad_id
        return subword_ids, attention_mask


if __name__ == "__main__":
    dataset = ScanDataset(
        "../data/eight_verbs/datasets/high_uniform.txt", "cpu", "scan", True
    )
