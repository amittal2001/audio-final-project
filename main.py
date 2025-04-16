from torch.backends.cudnn import deterministic

from src import evaluation as eval_module
from src import training

import argparse
import sys

from src.training import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Main entry point for training or evaluating TinySpeech models."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--no-seed", action="store_true",
                              help="If set, do not set a random seed (nondeterministic training)")
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed value for reproducible training (ignored if --no-seed is set)")
    train_parser.add_argument("--name", type=str, default="",
                              help="Custom name suffix for this run")

    # Subparser for evaluation
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--weights", type=str, required=True,
                             help="Path to the saved model weights")
    eval_parser.add_argument("--model", type=str, required=True,
                             choices=["TinySpeechX", "TinySpeechY", "TinySpeechZ", "TinySpeechM"],
                             help="Model architecture to use")
    eval_parser.add_argument("--file", type=str,
                             help="Path to a single audio file for evaluation")
    eval_parser.add_argument("--label", type=str,
                             help="True label for the audio file (optional)")

    args = parser.parse_args()

    if args.command == "train":
        new_argv = [sys.argv[0]]
        if args.no_seed:
            new_argv.append("--no-seed")
        if args.name:
            new_argv.extend(["--name", args.name])
        sys.argv = new_argv

        training.train_models(name_suffix=args.name, deterministic=args.no_seed)

    elif args.command == "eval":
        # Adjust sys.argv for evaluation.
        new_argv = [sys.argv[0], "--weights", args.weights, "--model", args.model]
        if args.file:
            new_argv.extend(["--file", args.file])
        if args.label:
            new_argv.extend(["--label", args.label])
        sys.argv = new_argv

        eval_module.main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
