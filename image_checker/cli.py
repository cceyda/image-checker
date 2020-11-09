import argparse
import logging
import logging.config
import os
import json
from .checker import checker_batch
from .iterators import folder_iterator


def main():
    parser = argparse.ArgumentParser(description="Check a folder of images for broken/misidentified images")

    parser.add_argument("-p", "--path", help="Path for folder to be checked", default="./", type=str)
    parser.add_argument(
        "-b", "--batch_size", default=50, help="Number of files checked per iteration (Recommend <100)", type=int
    )
    parser.add_argument(
        "-g", "--device_id", default=0, help="Gpu ID", type=int
    )
    
    #parser.add_argument("-fetch", "--prefetch", default=2, help="Number of batches prefetched", type=int)

    parser.add_argument(
        "-ext",
        "--extensions",
        default="jpeg,jpg,png",
        help="(comma delimited) list of extentions to test for (only types supported by DALI)",
        type=str,
    )
    parser.add_argument("-l", "--log_conf", default="logging_config.json", help="Config file path", type=str)
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-c", "--use_cpu", action="store_true")
    args = parser.parse_args()
    args = vars(args)
    
    args["prefetch"]=2
    if args["use_cpu"]:
        args["device"]="cpu"
    else: 
        args["device"]="mixed"
    try:
        args["extensions"] = args["extensions"].split(",")
    except:
        print("extensions should be comma delimited")

    # args = {
    #     "path": "/mnt/data/images/main/",
    #     "batch_size": 50,
    #     "prefetch": 2,
    #     "debug": True,
    #     "extensions": ["jpeg", "jpg", "png"],
    #     "recursive":False,
    #     "log_conf":"logging_config.json",
    #     "device":"mixed",
    #     "device_id":0
    #
    # }

    print(args)
    
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),args["log_conf"])
    try:
        with open(log_file_path) as f:
            config_dict = json.load(f)
            if args["debug"]:
                config_dict["loggers"]["image_checker"]["level"]="DEBUG"
        logging.config.dictConfig(config_dict)
    except Exception as e:
        raise Exception(f"Problem with logfile {e}")
    
    ds = folder_iterator(args["path"], args["extensions"], args["recursive"])
    checker_batch(ds, args)