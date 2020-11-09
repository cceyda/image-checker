import logging
import os

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
from more_itertools import chunked
from nvidia.dali.pipeline import Pipeline

log = logging.getLogger("image_checker")

# TODO
# shard_id = torch.cuda.current_device()
# num_shards = torch.cuda.device_count()
shard_id = 0
num_shards = 0
local_rank = 0

class DaliChecker:
    def __init__(self, batch_size, prefetch=2, device="mixed", device_id=0):
        log.debug("making checker")
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.device = device
        self.device_id = device_id
        self.make_pipe()
        self.pipe.build()

    def make_pipe(self):
        log.debug("making pipe")
        self.pipe = Pipeline(batch_size=self.batch_size, num_threads=2, device_id=self.device_id, prefetch_queue_depth=self.prefetch)
        with self.pipe:
            self.files = fn.external_source()
            images = fn.image_decoder(self.files, device=self.device)
            self.pipe.set_outputs(images)

    def feed(self, images):
        self.pipe.feed_input(self.files, images)
