#!/usr/bin/env python3
"""Compatibility entrypoint for DIME training.

Delegates to `shivangi_train.py` so existing commands like
`python train.py ...` keep working.
"""

from shivangi_train import main


if __name__ == "__main__":
    raise SystemExit(main())
