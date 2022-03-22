#! /usr/bin/env python3

import sys
sys.path.append('core')

# from torchinfo import summary
# from torchviz import make_dot

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    raft = RAFT(args)
    model = torch.nn.DataParallel(raft)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # summary(model, input_size=2 * [(1, 3, 640, 480)], device = 'cuda')

    # with torch.no_grad():
    #     x = torch.zeros(1, 3, 640, 480, dtype=torch.float, requires_grad=False, device="cuda")
    #     # dot = make_dot(model(image1=x, image2=x, iters=torch.tensor(20), test_mode=torch.tensor(True)),
    #     #                 params=dict(list(model.named_parameters())),
    #     #                 show_attrs=True, show_saved=True)

    #     flow_low, flow_up = raft.forward(x, x, iters=torch.tensor(20), test_mode=torch.tensor(True))
    #     dot = make_dot(flow_up,
    #                     params=dict(list(model.named_parameters())),
    #                     show_attrs=True, show_saved=True)

    #     dot.render("viz_raft", format="png")

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=torch.tensor(4), test_mode=torch.tensor(True))

            # make_dot((flow_low, flow_up)).render("viz_raft", format="png")

            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
