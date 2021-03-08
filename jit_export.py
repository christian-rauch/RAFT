#! /usr/bin/env python3

import sys
sys.path.append('core')

import torch
from raft import RAFT

from collections import OrderedDict

import argparse
from argparse import Namespace

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="", help="models/raft-small.pth")
    parser.add_argument('--iterations', type=int, default=20, help="number of iterations")
    args = parser.parse_args()

    # example images to propagate dimensions
    example1 = torch.rand(1, 3, 640, 480)
    example2 = torch.rand(1, 3, 640, 480)
    if torch.cuda.is_available():
        example1 = example1.cuda()
        example2 = example2.cuda()

    data_parallel = False

    # small RAFT network, a.k.a. RAFT-S
    model_args = Namespace(small=True, mixed_precision=False, alternate_corr=False)
    weights = torch.load("models/raft-small.pth")

    if data_parallel:
        net = torch.nn.DataParallel(RAFT(model_args))
        net.load_state_dict(weights)
    else:
        net = RAFT(model_args)
        # remove 'model' prefix from data-parallel weights
        # source: https://learnopencv.com/optical-flow-using-deep-learning-raft/
        new_weights = OrderedDict()
        for name in weights:
            # create new name and update new model
            new_name = name[7:]
            new_weights[new_name] = weights[name]
        net.load_state_dict(new_weights)

    if torch.cuda.is_available():
        net = net.cuda()

    # export TorchScript vit JIT trace
    traced_script_module = torch.jit.trace(net, example_inputs = (example1, example2, torch.tensor(args.iterations), torch.tensor([]), torch.tensor(True), torch.tensor(True)))
    traced_script_module.save("RAFT_small_iter"+str(args.iterations)+".pt")
