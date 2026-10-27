"""Prepare a data bundle using the existing Lambda0 CLI."""
import sys
from lambda0.__main__ import main


if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
