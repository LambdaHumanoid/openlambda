"""Train or preflight an experiment; arguments and working directory are preserved."""
import sys
from lambda0.__main__ import main


if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
