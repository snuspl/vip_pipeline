from pathlib import Path
import random
import sys

from fire import Fire
from munch import Munch

import time
import torch
import numpy as np

from config import config, debug_options
from datasets import get_iterator
from util import wait_for_key, suppress_stdout
from train import train
from evaluate import evaluate
from infer import infer

from ckpt import get_model_ckpt

import os 
class Cli:
    def __init__(self, **kwargs):
        self.defaults = config
        self.debug = debug_options

    def _default_args(self, **kwargs):
        args = self.defaults
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        args.update(resolve_paths(config))
        args.update(fix_seed(args))
        args.update(get_device(args))

        return Munch(args)

    def check_dataloader(self, **kwargs):
        args = self._default_args(**kwargs)

        iters, vocab = get_iterator(args)
        for batch in iters['train']:
            import ipdb; ipdb.set_trace()  # XXX DEBUG

    def train(self, **kwargs):
        args = self._default_args(**kwargs)

        train(args)

        wait_for_key()

    def evaluate(self, **kwargs):
        args = self._default_args(**kwargs)

        evaluate(args)

        wait_for_key()

    def load_model(self, **kwargs):
        args = self._default_args(**kwargs)

        args.question = "init"
        args.vid = "init"
        args.ckpt_name = "vqa_model.pickle"

        args, model, iters, vocab, ckpt_available = get_model_ckpt(args)

        return [args, model, iters, vocab, ckpt_available]

    def infer(self, data, question, vid):
        #question="Who are the actors in the Friends?"
        #vid="s01e22_02"
        data[0].question = question
        data[0].vid = vid + '_122' # for alignment
        
        #print("Question :", question)
        #print("VideoID :", vid)

        ans = infer(data)

        return ans


def resolve_paths(config):
    paths = [k for k in config.keys() if k.endswith('_path')]
    res = {}
    print(paths)
    for path in paths:
        res[path] = Path(config[path])

    print(res)
    return res


def fix_seed(args):
    if 'random_seed' not in args:
        args['random_seed'] = 0
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    return args


def get_device(args):
    if hasattr(args, 'device'):
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {'device': device}

class VQAModel():
    def __init__(self):
        cli = Cli()
        self.data = cli.load_model()

    def predict(self, input_data):
        data = self.data
        question = input_data["question"]
        vid = input_data["vid"]

        data[0].question = question
        data[0].vid = vid + '_153' # for alignment
            
        answer = infer(data)

        return answer

