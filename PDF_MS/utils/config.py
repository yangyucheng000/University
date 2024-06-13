import json5
from easydict import EasyDict


def get_config_from_json(json_file):
    # parse the configurations from the configs json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json5.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(args):
    config = get_config_from_json(args.config)
    config.commit_id = args.id
    config.time_stamp = args.ts
    config.directory = args.dir
    return config