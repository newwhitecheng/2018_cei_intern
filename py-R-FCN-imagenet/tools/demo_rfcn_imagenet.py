#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import pdb
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
    'accordion','airplane','ant','antelope','apple',
    'armadillo','artichoke','axe','baby bed','backpack',
    'bagel','balance beam','banana','band aid','banjo',
    'baseball','basketball','bathing cap','beaker','bear',
    'bee','bell pepper','bench','bicycle','binder','bird',
    'bookshelf','bow tie','bow','bowl','brassiere','burrito',
    'bus','butterfly','camel','can opener','car','cart','cattle',
    'cello','centipede','chain saw','chair','chime','cocktail shaker',
    'coffee maker','computer keyboard','computer mouse','corkscrew',
    'cream','croquet ball','crutch','cucumber','cup or mug','diaper',
    'digital clock','dishwasher','dog','domestic cat','dragonfly',
    'drum','dumbbell','electric fan','elephant','face powder',
    'fig','filing cabinet','flower pot','flute','fox','french horn',
    'frog','frying pan','giant panda','goldfish','golf ball','golfcart',
    'guacamole','guitar','hair dryer','hair spray','hamburger','hammer',
    'hamster','harmonica','harp','hat with a wide brim','head cabbage',
    'helmet','hippopotamus','horizontal bar','horse','hotdog','iPod',
    'isopod','jellyfish','koala bear','ladle','ladybug','lamp','laptop',
    'lemon','lion','lipstick','lizard','lobster','maillot','maraca',
    'microphone','microwave','milk can','miniskirt','monkey','motorcycle',
    'mushroom','nail','neck brace','oboe','orange','otter','pencil box',
    'pencil sharpener','perfume','person','piano','pineapple',
    'ping-pong ball','pitcher','pizza','plastic bag','plate rack',
    'pomegranate','popsicle','porcupine','power drill','pretzel',
    'printer','puck','punching bag','purse','rabbit','racket','ray',
    'red panda','refrigerator','remote control','rubber eraser',
    'rugby ball','ruler','salt or pepper shaker','saxophone','scorpion',
    'screwdriver','seal','sheep','ski','skunk','snail','snake','snowmobile',
    'snowplow','soap dispenser','soccer ball','sofa','spatula','squirrel',
    'starfish','stethoscope','stove','strainer','strawberry','stretcher',
    'sunglasses','swimming trunks','swine','syringe','table','tape player',
    'tennis ball','tick','tie','tiger','toaster','traffic light','train',
    'trombone','trumpet','turtle','tv or monitor','unicycle','vacuum','violin',
    'volleyball','waffle iron','washer','water bottle','watercraft',
    'whale','wine bottle','zebra')

NETS = {'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_ohem_iter_100001.caffemodel'),
'GoogleNet': ('GoogleNet',
                  'GoogleNet_rfcn_ohem_iter_230000.caffemodel'),
'CaffeNet': ('CaffeNet',
                  'CaffeNet_rfcn_ohem_iter_10000.caffemodel')

}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='CaffeNet')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'imagenet_models',
                              NETS[args.demo_net][1]) 
    #pdb.set_trace()

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    print caffe.TEST
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    
    for i in xrange(2):
        _, _= im_detect(net, im)

    
    im_names = ['000456.jpg', '000542.jpg', '001150.jpg','001763.jpg', '004545.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
