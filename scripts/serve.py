"""Serve Lambda0 using the existing inference implementation."""
import sys
from lambda0.__main__ import main


if __name__ == "__main__":
    main(["serve", *sys.argv[1:]])
