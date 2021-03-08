#! /usr/bin/env python3

import sys
sys.path.append('core')

import argparse
import os
import glob
import torch

from utils.utils import InputPadder

from demo import load_image, viz

import time

DEVICE = 'cuda'

def demo(args):
    model = torch.jit.load(args.model)

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # call the forward method on the TorchScript model
            # all static values of the original model, such as 'iters', were fixed during the JIT tracing and are ignored here
            tstart = time.time()
            flow_low, flow_up = model(image1, image2, torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))
            print("inference", time.time()-tstart, "s")
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="RAFT_small_iter20.pt")
    parser.add_argument('--path', type=str, help="dataset for evaluation")
    args = parser.parse_args()

    demo(args)
