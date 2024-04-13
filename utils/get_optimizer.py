import os
import sys

sys.path.append(os.path.abspath("."))
from utils.import_file import import_file


def get_optimizer(optimizer: str):

    base_dir = os.path.join(os.getcwd(), "optimizers")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{optimizer}.py"),
    )
    return all_modules.get_optim()
