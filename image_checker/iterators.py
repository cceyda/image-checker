import os
import numpy as np
import logging
from pathlib import Path

log = logging.getLogger("image_checker")


def folder_iterator(image_folder,extensions, recursive):
    files = os.scandir(image_folder)
    yield from file_iterator(files,extensions,recursive)

def file_iterator(files,extensions,recursive):
    for fil in files:
        if isinstance(fil, str):
            fil = Path(fil)
        if fil.is_file():
            if any([fil.name.endswith(x) for x in extensions]):
                if isinstance(fil, Path):
                    image_path = str(fil.resolve())
                else:
                    image_path = fil.path
                try:
                    f = open(image_path, "rb")
                    image = np.frombuffer(f.read(), dtype=np.uint8)
                    yield image, image_path
                except Exception as e:
                    log.error(f"Couldn't read file: {image_path}, Err:{e}")
        elif recursive and fil.is_dir():
            yield from folder_iterator(fil.path,extensions,recursive)