"""One lazy CLI for source checkouts and installed distributions."""
import argparse
import importlib
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {"prepare": "lambda0.data.prepare", "train": "lambda0.training.runner",
                "serve": "lambda0.inference.server", "doctor": "lambda0.runtime.doctor"}
    parser = argparse.ArgumentParser(prog="lambda0", description="Prepare data, train, serve, or inspect an environment.")
    parser.add_argument("command", choices=commands)
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    importlib.import_module(commands[command]).main(args[1:])


if __name__ == "__main__":
    main()
