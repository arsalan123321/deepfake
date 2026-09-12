"""
Wraps kaggle/notebook/worker.py into a single-cell worker.ipynb so it can be
pushed with `kaggle kernels push`. Keeping worker.py as plain Python for easy
editing/diffing; this script is the only place that touches notebook JSON.
"""
import json
from pathlib import Path

SRC = Path("kaggle/notebook/worker.py")
OUT = Path("kaggle/notebook/worker.ipynb")

def main():
    code = SRC.read_text()
    nb = {
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=2))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
