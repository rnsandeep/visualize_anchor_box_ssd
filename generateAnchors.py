from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
import numpy as np


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

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))

                average = (s_k + s_k_prime)/2
                mean += [cx, cy, average/sqrt(1.5), average*sqrt(1.5)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output



voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [10, 5, 3, 1],
    'min_dim': 300,
    'steps': [32, 64, 100, 300],
    'min_sizes': [15, 15, 15, 15],
    'max_sizes': [60, 60, 60, 60],
    'aspect_ratios': [[1]], #[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


def drawBoxes(boxes, I):
    for i, box in enumerate(boxes):
        color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        thickness = 2
        box = [i*300 for i in box] # denormalize the box
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        I = cv2.rectangle(I, start_point, end_point, color, thickness)
    return I


pbox = PriorBox(voc)

boxes = pbox.forward() # get prior boxes.

# convert centered boxes to xmin, ymin, xmax, ymax.
boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2
boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

boxes_numpy = boxes.squeeze()



import cv2, sys, os

I = cv2.imread(sys.argv[1])
I = cv2.resize(I, (300, 300))

I = drawBoxes(boxes_numpy, I)  # overlay boxes on Image.

output_path = 'output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

cv2.imwrite(os.path.join(output_path, os.path.basename(sys.argv[1])), I)
