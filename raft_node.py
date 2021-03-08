#! /usr/bin/env python3

import sys
sys.path.append('core')

import torch
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

import collections
import cv2
from argparse import Namespace
import time


bridge = CvBridge()
imgs = collections.deque(maxlen=2)

last_img = None


def msg_to_tensor(msg_compr_img):
    global imgs, last_img

    img_cv = bridge.compressed_imgmsg_to_cv2(msg_compr_img, desired_encoding='passthrough')

    img_cv = cv2.resize(img_cv, None, fx=0.5, fy=0.5)

    last_img = img_cv

    tensor = torch.from_numpy(img_cv).permute(2, 0, 1).float().unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def on_compr_img(compr_img):
    imgs.append(msg_to_tensor(compr_img))

def process(event):
    if len(imgs)<2:
        return

    with torch.no_grad():
        padder = InputPadder(imgs[0].shape)
        image1, image2 = padder.pad(imgs[0], imgs[1])

        tstart = time.time()
        flow_ups = model(image1, image2, iters=torch.tensor(10), test_mode=torch.tensor(False))
        tend = time.time()
        print("inference", tend-tstart,"s")

        flo = flow_ups[-1][0].permute(1,2,0).cpu().detach().numpy()
        flo = flow_viz.flow_to_image(flo)

        cv2.imshow("colour", last_img)
        cv2.imshow('flow', flo/255.0)
        cv2.waitKey(1)


if __name__ == '__main__':

    model_args = Namespace(small=True, mixed_precision=False, alternate_corr=False)
    model = torch.nn.DataParallel(RAFT(model_args))
    model.load_state_dict(torch.load("models/raft-small.pth"))

    model = model.module
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    rospy.init_node('raft')

    rospy.Subscriber("/rgb/image_raw/compressed", CompressedImage, on_compr_img)

    rospy.Timer(rospy.Duration(0.01), process)

    try:
        rospy.spin()
    except rospy.exceptions.ROSTimeMovedBackwardsException:
        pass
