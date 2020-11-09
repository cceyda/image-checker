# Fast Image Integrity Checker
Check for corrupted JPEG,PNG (and more) images in bulk using GPU jpeg decoding powered by NVIDIA DALI!

Super fast compared to alternatives because it uses GPU decoding and checks in batches.

# Requirements
Depending on your cuda version install:
[NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html#id1)
If it fails try `pip3 install –upgrade pip`

# Usage

`pip3 install image-checker`

or

`git clone https://github.com/cceyda/image-checker.git`

## CLI
There is a CLI that takes a folder `--path` to scan for images with the given extensions `--ext`  
outputs an `error.log`. Format of log file can be modified by `--log_conf` error handler using python logging [example](/master/dali_image_checker/logging_config.json). If you don't want to use gpu provide `--use_cpu` flag. Use `--recursive` to 
traverse sub-folders.

`image-check --path /mnt/data/dali_test/corrupt --recursive`

```bash
usage: image-checker [-h] [-p PATH] [-b BATCH_SIZE] [-g DEVICE_ID]
                        [-ext EXTENSIONS] [-l LOG_CONG] [-r] [-d] [-c]

Check a folder of images for broken/misidentified images

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path for folder to be checked
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of files checked per iteration (Recommend <100)
  -g DEVICE_ID, --device_id DEVICE_ID
                        Gpu ID
  -ext EXTENSIONS, --extensions EXTENSIONS
                        (comma delimited) list of extentions to test for (only types supported by DALI)
  -l LOG_CONG, --log_conf LOG_CONF
                        Config file path
  -r, --recursive
  -d, --debug
  -c, --use_cpu
```

## Code

```python
from image_checker.checker import checker_batch,checker_single
from image_checker.iterators import folder_iterator

args = {
     "path": "/mnt/data/images/main/",
     "batch_size": 50,
     "prefetch": 2,
     "debug": False,
     "extensions": ["jpeg", "jpg", "png"],
     "recursive":False,
     "log_cong":"logging_config.json",
     "device":"mixed",
     "device_id":0
 }

ds = folder_iterator(args["path"], args["extensions"], args["recursive"])
bad_files=checker_batch(ds, args)

```

# FAQ

- What kind of corrupted images will this catch?
    - Images that can't be decoded by DALI.
    - GIFs pretending to be JPEGs (with a jpg,jpeg extension)
    - (Won't catch) files that can't be opened (TODO)
    - (Won't catch) empty image files
    
- Supported image types?

Same as [DALI supported formats](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html?highlight=supported%20image#nvidia.dali.ops.ImageDecoder): JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000. Please note that GPU acceleration for JPEG 2000 decoding is only available for CUDA 11.
    
- What is batch_size & prefetch?

DALI works with a batching+prefetching system. So batch_size * prefetch number of images are read at a time. If there is a corrupted file in the batch that batch is rechecked 1-by-1. So keep batch_size reasonable (0<100)


- Package versioning follows dali for major.minor (since it heavily depends on it), patch is this packages version changes.

# Alternatives
[check-media-integrity](https://github.com/ftarlao/check-media-integrity): supports more types but uses PIL thus slow.

[Bad Peggy](https://github.com/llaith-oss/BadPeggy): Checks JPEG images, maybe detects more types of errors than this one.
