from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
import numpy as np

import os
from xml.dom import minidom
os.environ['GLOG_minloglevel'] = '2'
import  os, sys, cv2
import argparse
import shutil
import sys ,os


from drawBoxes import read_xml

import os, shutil

def files_with_ext(mypath, ext):
    files = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and os.path.splitext(os.path.join(mypath, f))[1] == ext]
    return files


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]

                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
#               mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))

                average = (s_k + s_k_prime)/2
                mean += [cx, cy, average/sqrt(1.5), average*sqrt(1.5)]

                # rest of aspect ratios
#                for ar in self.aspect_ratios[k]:
#                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
#                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output



voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [0, 30, 30, 45, 45, 45], #[45, 45, 45, 45, 45, 45],
    'max_sizes': [30, 45, 45, 90, 90, 90], #[75, 75, 75, 75, 75, 75],
    'aspect_ratios': [[1]], #[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


pbox = PriorBox(voc)

boxes = pbox.forward()
#boxes_numpy = boxes.squeeze()

boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2
boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

boxes_priors = boxes*300


from box_utils import match



import cv2, sys, os



images = files_with_ext(sys.argv[1], '.JPG')
xmls = files_with_ext(sys.argv[2], '.xml')


scores = 0
for image in images:
    print(image)
    image_name = image
    xml_name = os.path.join(sys.argv[2], os.path.basename(image_name).replace('.JPG', '.xml'))
    objects, width, height = read_xml(xml_name)

    boxes = []
    for obj in objects:
        cboxes = objects[obj]
        newboxes = [[int(i) for i in box] for box in cboxes]
        boxes += newboxes
    match_score = match(0.9, torch.FloatTensor(boxes), boxes_priors)
    scores += match_score
    print(match_score)

print("total scores:", scores)
    

