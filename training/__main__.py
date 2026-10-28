"""Entry points for portable preparation and bounded native policy training."""
import argparse
import importlib
import sys

COMMANDS = {
    "run": "training.scripts.run",
    "prepare": "training.scripts.prepare",
    "pipeline": "training.scripts.pipeline",
    "batch": "training.scripts.batch",
    "evaluate": "training.evaluate.policy",
    "export": "training.evaluate.export",
    "score-clean": "training.evaluate.score_clean_reference",
    "bootstrap": "training.scripts.bootstrap",
    "setup-runtime": "training.scripts.setup_runtime",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    if not args or args[0] in ("-h", "--help"):
        parser.parse_args(args)
        return
    command = parser.parse_args(args[:1]).command
    module = importlib.import_module(COMMANDS[command])
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *args[1:]]
        return module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
