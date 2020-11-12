import gc
import logging
import os
import warnings
import numpy as np

# import torch
from more_itertools import chunked, grouper
from tqdm import tqdm
from pathlib import Path
from .DaliChecker import DaliChecker
from .iterators import file_iterator

# TODO handle warnings
# TODO pretty tqdm
# TODO better file list logging
# TODO exact good image count (currently the batch is padded)

log = logging.getLogger("image_checker")
log_err = logging.getLogger("image_checker.errors")


def batch(iterator, batch_size):
    for x in grouper(iterator, batch_size, (np.zeros([1]), "fill.jpg")):
        z = list(zip(*x))
        yield list(z[0]), list(z[1])


def checker_batch(ds, args):
    if args["batch_size"]==1 and args["prefetch"]==1:
        return checker_single(ds,args)
    batched_iterator = batch(ds, args["batch_size"])
    has_more = True
    baddies = []
    pbar = tqdm(total=0, desc=f"Good images (±{str(args['batch_size'])})", leave=True)
    pbar_bad = tqdm(total=0, desc="Bad images", leave=True)

    checker = DaliChecker(args["batch_size"], args["prefetch"],args["device"],args["device_id"])

    def loop(checker):
        nonlocal has_more
        try:
            # prefetch
            image_paths = []
            for _ in range(args["prefetch"]):
                try:
                    images, paths = next(batched_iterator)
                    image_paths.extend(paths)
                    checker.feed(images)
                except StopIteration:
                    has_more = False
            if image_paths:    
                checker.pipe.run()
                pbar.update(len(image_paths))
            return None
        except Exception as e:
            #log.debug(e)
            #log.debug(image_paths)
            log.debug("Found bad files in batch")
            log.debug("cleaning old pipe")
            del checker
            gc.collect()
            #             torch.cuda.empty_cache()
            log.debug("end clean")
            return image_paths

    while has_more:
        potential_bad_paths = loop(checker)
        if potential_bad_paths:
            log.debug("Rescanning batch")
            potential_bad_paths = list(map(Path, potential_bad_paths))
            ds = file_iterator(potential_bad_paths, args["extensions"], False)
            bad_paths = checker_single(ds,args, pbar, pbar_bad)
            baddies.extend(bad_paths)
            for x in bad_paths:
                log_err.error(x)
            log.debug("Continuing Scan")
            checker = DaliChecker(args["batch_size"], args["prefetch"],args["device"],args["device_id"])
    log.info("End of Scan")
    log.info(f"Found {len(baddies)} bad files")
    return baddies


def checker_single(ds,args, pbar=None, pbar_bad=None):
    batched_iterator = batch(ds, 1)
    has_more = True
    bad_paths = []
    if pbar is None:
        pbar = tqdm(total=0, desc="Good images", leave=True)
    if pbar_bad is None:
        pbar_bad = tqdm(total=0, desc="Bad images", leave=True)

    checker = DaliChecker(1, 1,args["device"],args["device_id"])

    def loop(checker):
        nonlocal has_more
        try:
            # prefetch
            image_paths = []
            for _ in range(1):
                try:
                    images, paths = next(batched_iterator)
                    image_paths.extend(paths)
                    checker.feed(images)
                except StopIteration:
                    has_more = False
            if image_paths:   
                checker.pipe.run()
                pbar.update(len(image_paths))
            return None
        except Exception as e: 
            bad_paths.extend(image_paths)
            log.debug(e)
            log.debug(image_paths)
            pbar_bad.update(len(image_paths))
            log.debug("Found bad file")
            log.debug("cleaning old pipe")
            del checker
            gc.collect()
            #             torch.cuda.empty_cache()
            log.debug("end clean")
            return True

    while has_more:
        reset = loop(checker)
        if reset is not None:
            checker = DaliChecker(1, 1,args["device"],args["device_id"])

    del checker
    gc.collect()
    return bad_paths
