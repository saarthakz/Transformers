import os
import sys

sys.path.append(os.path.abspath("."))
from utils.import_file import import_file


def get_model_arch(model_arch: str):

    base_dir = os.path.join(os.getcwd(), "model_archs")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{model_arch}.py"),
    )
    return all_modules.Model
