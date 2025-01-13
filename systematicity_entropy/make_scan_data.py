import argparse
import math
import re
from pathlib import Path

from systematicity_entropy.PCFG import PCFG

ATOMS = {
    "walk": "WALK",
    "sprint": "SPRINT",
    "crawl": "CRAWL",
    "squat": "SQUAT",
    "lunge": "LUNGE",
    "jump": "JUMP",
    "run": "RUN",
    "look": "LOOK",
    "turn left": "LTURN",
    "turn right": "RTURN",
    "turn opposite left": "LTURN LTURN",
    "turn opposite right": "RTURN RTURN",
    "turn around right": "RTURN RTURN RTURN RTURN",
    "turn around left": "LTURN LTURN LTURN LTURN",
}

UNARIES = re.compile(
    "(.*)(walk|look|run|jump|sprint|crawl|squat|lunge) (?:(opposite|around) )?(left|right)(.*)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCAN data")
    parser.add_argument(
        "--n", type=int, default=1000, help="The number of samples to generate"
    )
    parser.add_argument(
        "--grammar",
        type=Path,
        default=Path("./scan_cfg.cfg"),
        help="Path to the PCFG to generate data from",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Path to where you want to save the generated data",
    )
    parser.add_argument(
        "--exclude_path",
        type=Path,
        default=None,
        help="Path to where you want to save the generated data",
    )
    parser.add_argument(
        "--split", action="store_true", help="Creates a disjoint train and test split"
    )
    return parser.parse_args()


def translate(instructions: str) -> str:
    if " and " in instructions:
        return " ".join(map(translate, instructions.split(" and ")))
    if " after " in instructions:
        return " ".join(reversed(list(map(translate, instructions.split(" after ")))))
    output: str = instructions
    while match := UNARIES.fullmatch(output):
        if match[3] is None:
            direction = match[4][0].upper()
            output = f"{direction}TURN {match[2]}"
        elif match[3] == "opposite":
            output = f"turn opposite {match[4]} {match[2]}"
        elif match[3] == "around":
            direction = match[4][0].upper()
            output = " ".join(4 * [f"{direction}TURN {match[2]}"])
        output = f"{match[1]}{output}{match[5]}"
    for atom, value in ATOMS.items():
        output = output.replace(atom, value)

    if output.endswith(" twice"):
        output = output.rstrip(" twice")
        output = f"{output} {output}"
    elif output.endswith(" thrice"):
        output = output.rstrip(" thrice")
        output = f"{output} {output} {output}"
    return output


def main():
    args = parse_args()
    print("SCAN dataset generator, arguments:")
    print(vars(args))
    pcfg = open(args.grammar, "r").read()
    grammar = PCFG.fromstring(pcfg)
    samples = set()
    for sentence in grammar.generate(args.n):
        output = translate(sentence)
        sample = f"IN: {sentence} OUT: {output}\n"  # Match the original SCAN format.
        samples.add(sample)

    samples = list(samples)
    num_samples = len(samples)
    print(f"Generated {num_samples} unique samples from {args.n} iterations...")
    if args.exclude_path:
        removed = 0
        with open(args.exclude_path, "r") as f:
            exclusion_list = f.readlines()
            for ex in exclusion_list:
                if ex in samples:
                    samples.remove(ex)
                    removed += 1
        print(f"Removed {removed} samples..")
    if args.split:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        training_ratio = 0.8
        split_point = math.floor(training_ratio * num_samples)
        train_samples = samples[:split_point]
        test_samples = samples[split_point:]
        with open(args.save_path / "train.txt", "w") as f:
            f.writelines(train_samples)
        with open(args.save_path / "test.txt", "w") as f:
            f.writelines(test_samples)
    else:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_path, "w") as f:
            f.writelines(samples)


if __name__ == "__main__":
    main()
