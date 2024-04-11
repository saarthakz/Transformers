import os
import sys

sys.path.append(os.path.abspath("."))
from utils.import_file import import_file


def get_dataset(dataset_name: str, input_res: list[int] = [32, 32]):

    base_dataset_dir = os.path.join(os.getcwd(), "datasets")
    all_modules = import_file(
        "",
        path=os.path.join(base_dataset_dir, f"{dataset_name}.py"),
    )
    return all_modules.get_dataset(input_res)
