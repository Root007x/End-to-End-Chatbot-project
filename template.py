import os
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO, format='[%(asctime)s:%(message)s]')


list_files = [
    "src/__init__.py",
    "src/helper.py",
    "setup.py",
    "app.py",
    "research/experiment.ipynb"
]

for filepath in list_files:
    file_path = Path(filepath)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating Directory : {file_dir} for the file {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating empty file {file_name}")

    else:
        logging.info(f"{file_name} is already exists")



    