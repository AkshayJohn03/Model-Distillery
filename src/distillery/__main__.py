"""``python -m distillery`` entrypoint."""

import sys

from distillery.cli import main

if __name__ == "__main__":
    sys.exit(main())
