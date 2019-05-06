# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------

import os, sys
import numpy as np
import argparse
import pprint
import time
import csv
csv.field_size_limit(sys.maxsize)
import json
import base64
import _pickle as cPickle
import torch
import pdb

from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.roi_layers import nms
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import demo


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
MIN_BOXES = 10
MAX_BOXES = 100


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models/")
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--cpu', dest='cuda',
                        help='whether use CUDA',
                        action='store_false')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=9771, type=int)
    parser.add_argument('--out', dest='outfile',
                        help='output file name',
                        default='coco_train_faster-rcnn_res101_ls.tsv', type=str)

    args = parser.parse_args()
    return args


def load_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    if split_name == 'coco_val2014':
        with open('./data/coco/annotations/instances_val2014.json') as f:
            data = json.load(f)
            for item in data['images']:
                image_id = int(item['id'])
                filepath = os.path.join('./data/coco/images/val2014/', item['file_name'])
                split.append((filepath,image_id))
    elif split_name == 'coco_test2015':
        with open('./data/coco/annotations/image_info_test2015.json') as f:
            data = json.load(f)
            for item in data['images']:
                image_id = int(item['id'])
                filepath = os.path.join('./data/coco/images/test2015/', item['file_name'])
                split.append((filepath,image_id))
    elif split_name == 'genome':
        with open('./data/visualgenome/image_data.json') as f:
            for item in json.load(f):
                image_id = int(item['image_id'])
                filepath = os.path.join('./data/visualgenome/', item['url'].split('rak248/')[-1])
                split.append((filepath,image_id))
    elif split_name == 'flickr30k':
        image_dir = './data/flickr30k/images/'
        for file_name in os.listdir(image_dir):
            filepath = os.path.join(image_dir, file_name)
            image_id = int(file_name.split('.')[0])
            split.append((filepath,image_id))
    elif split_name == 'referit_train':
        image_dir = './data/referit/ImageCLEF/images/'
        image_list = load_int_list('./data/referit/split/referit_train_imlist.txt')
        split = make_split(image_dir, image_list)
    elif split_name == 'referit_val':
        image_dir = './data/referit/ImageCLEF/images/'
        image_list = load_int_list('./data/referit/split/referit_val_imlist.txt')
        split = make_split(image_dir, image_list)
    elif split_name == 'referit_trainval':
        image_dir = './data/referit/ImageCLEF/images/'
        image_list = load_int_list('./data/referit/split/referit_trainval_imlist.txt')
        split = make_split(image_dir, image_list)
    elif split_name == 'referit_test':
        image_dir = './data/referit/ImageCLEF/images/'
        image_list = load_int_list('./data/referit/split/referit_test_imlist.txt')
        split = make_split(image_dir, image_list)
    elif split_name == 'clevr_train':
        image_dir = './data/CLEVR/CLEVR_v1.0/images/train/'
        split = []
        for file_name in os.listdir(image_dir):
            filepath = os.path.join(image_dir, file_name)
            image_id = int(file_name.split('.')[0].split('_')[-1])
            split.append((filepath , image_id))
    elif split_name == 'clevr_val':
        image_dir = './data/CLEVR/CLEVR_v1.0/images/val/'
        split = []
        for file_name in os.listdir(image_dir):
            filepath = os.path.join(image_dir, file_name)
            image_id = int(file_name.split('.')[0].split('_')[-1])
            split.append((filepath , image_id))
    elif split_name == 'clevr_test':
        image_dir = './data/CLEVR/CLEVR_v1.0/images/test/'
        split = []
        for file_name in os.listdir(image_dir):
            filepath = os.path.join(image_dir, file_name)
            image_id = int(file_name.split('.')[0].split('_')[-1])
            split.append((filepath , image_id))
    elif split_name == 'openimages_train':
        image_dir = './data/openimages/train/'
        image_list = load_openimage_vrd_list('./data/openimages/challenge-2018-train-vrd.csv')
        split = make_split(image_dir, image_list)
    elif split_name == 'openimages_challenge':
        image_dir = './data/openimages/challenge2018_test/'
        file_list = os.listdir(image_dir)
        image_list = [file.split('.')[0] for file in file_list]
        split = make_split(image_dir, image_list)
    elif split_name == 'vrd_train':
        image_dir = './data/vrd/sg_dataset/sg_train_images/'
        file_list = os.listdir(image_dir)
        image_list = [file.split('.')[0] for file in file_list]
        path_list = [os.path.join(image_dir, file_name) for file_name in file_list]
        split = list(zip(path_list, image_list))
    elif split_name == 'vrd_test':
        image_dir = './data/vrd/sg_dataset/sg_test_images/'
        file_list = os.listdir(image_dir)
        image_list = [file.split('.')[0] for file in file_list]
        path_list = [os.path.join(image_dir, file_name) for file_name in file_list]
        split = list(zip(path_list, image_list))
    elif split_name == 'gqa':
        image_dir = './data/GQA/images/'
        file_list = os.listdir(image_dir)
        image_list = [file.split('.')[0] for file in file_list]
        path_list = [os.path.join(image_dir, file_name) for file_name in file_list]
        split = list(zip(path_list, image_list))
    elif split_name == 'referring_train':
        data_dir = './data/ReferringRelationships/'
        image_dirs = [os.path.join(data_dir, image_dir) for image_dir in ['VG_100K', 'VG_100K_2']]
        image_files = list(json.load(open(os.path.join(data_dir, 'train_image_metadata.json'))).keys())
        image_list = [int(file.split('.')[0]) for file in image_files]
        path_list = [os.path.join(image_dirs[0], file_name) if os.path.isfile(os.path.join(image_dirs[0], file_name))\
                     else os.path.join(image_dirs[1], file_name) for file_name in image_files]
        split = list(zip(path_list, image_list))
    elif split_name == 'referring_test':
        data_dir = './data/ReferringRelationships/'
        image_dirs = [os.path.join(data_dir, image_dir) for image_dir in ['VG_100K', 'VG_100K_2']]
        image_files = list(json.load(open(os.path.join(data_dir, 'test_image_metadata.json'))).keys())
        image_list = [int(file.split('.')[0]) for file in image_files]
        path_list = [os.path.join(image_dirs[0], file_name) if os.path.isfile(os.path.join(image_dirs[0], file_name)) \
                         else os.path.join(image_dirs[1], file_name) for file_name in image_files]
        split = list(zip(path_list, image_list))
    elif split_name == 'tdiuc_train':
        image_dir = './data/TDIUC/Images/train2014/'
        file_list = os.listdir(image_dir)
        image_list = [int(file.split('.')[0].split('_')[-1]) for file in file_list]
        path_list = [os.path.join(image_dir, file_name) for file_name in file_list]
        split = list(zip(path_list, image_list))
    elif split_name == 'tdiuc_val':
        image_dir = './data/TDIUC/Images/val2014/'
        file_list = os.listdir(image_dir)
        image_list = [int(file.split('.')[0].split('_')[-1]) for file in file_list]
        path_list = [os.path.join(image_dir, file_name) for file_name in file_list]
        split = list(zip(path_list, image_list))
    else:
        print('Unknown split')
        pdb.set_trace()

    return split


def make_split(image_dir, image_list):
    split = []
    for image_id in image_list:
        filepath = os.path.join(image_dir, '%s.jpg' % str(image_id))
        if not os.path.isfile(filepath):
            continue
        split.append((filepath,image_id))
    return split


def load_int_list(filename):
    with open(filename, 'r') as f:
        str_list = f.readlines()
    int_list = [int(s[:-1]) for s in str_list]
    return int_list


def load_openimage_vrd_list(filename):
    data_dir = '/data/openimages/'
    field_names = ['ImageID', 'LabelName1', 'LabelName2',
                   'XMin1', 'XMax1', 'YMin1', 'YMax1',
                   'XMin2', 'XMax2', 'YMin2', 'YMax2', 'RelationshipLabel']

    image_ids = []
    with open(filename) as file:
        reader = csv.DictReader(file, fieldnames=field_names)
        header = next(reader)

        for item in reader:
            image_ids.append(item['ImageID'])

    return list(set(image_ids))


def get_spatial(bboxes, w, h):
    scaled_x = bboxes[:, [0, 2]] / w
    scaled_y = bboxes[:, [1, 3]] / h
    scaled_w = scaled_x[:, 1] - scaled_x[:, 0]
    scaled_h = scaled_y[:, 1] - scaled_y[:, 0]

    scaled_spatials = torch.stack((scaled_x[..., 0], scaled_y[..., 0],
                                   scaled_x[..., 1], scaled_y[..., 1],
                                   scaled_w, scaled_h), -1)
    return scaled_spatials


def create_input_holders(args):

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    gt_boxes = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        gt_boxes = gt_boxes.cuda()
        num_boxes = num_boxes.cuda()
    return im_data, im_info, gt_boxes, num_boxes


def get_detections_from_im(net, im_file, image_id, input_holders, thresh=0.2):
    # im = cv2.imread(im_file)
    im_in = np.array(imread(im_file))

    if len(im_in.shape) == 2:
        im_in = im_in[:, :, np.newaxis]
        im_in = np.concatenate((im_in, im_in, im_in), axis=2)
    # rgb -> bgr
    im = im_in[:, :, ::-1]

    blobs, im_scales = demo._get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    input_holders[0].data.resize_(im_data_pt.size()).copy_(im_data_pt)
    input_holders[1].data.resize_(im_info_pt.size()).copy_(im_info_pt)
    input_holders[2].data.resize_(1, 1, 5).zero_()
    input_holders[3].data.resize_(1).zero_()

    # pdb.set_trace()
    det_tic = time.time()

    rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, rois_label, feat = net(*input_holders)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    pred_boxes = demo.adjust_bbox(boxes, bbox_pred, input_holders[1], im_scales,
                                  scores.shape[1], num_classes, args, cfg)

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()

    max_conf = torch.zeros(scores.shape[0])
    bboxes = torch.zeros((scores.shape[0], 4))
    for j in range(1, num_classes):
        cls_scores = scores[:, j].cpu()
        cls_boxes = pred_boxes.cpu() if args.class_agnostic else pred_boxes[:, j * 4:(j + 1) * 4].cpu()
        keep = nms(cls_boxes, cls_scores, cfg.TEST.NMS)
        bboxes[keep] = torch.where(cls_scores[keep].unsqueeze(1).repeat(1, 4) \
                                   > max_conf[keep].unsqueeze(1).repeat(1, 4), \
                                   cls_boxes[keep], bboxes[keep])
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = torch.nonzero(max_conf >= thresh).squeeze(1)
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, 0, True)[:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, 0, True)[:MAX_BOXES]

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': base64.b64encode(bboxes[keep_boxes].numpy()).decode('utf'),
        'features': base64.b64encode(feat[keep_boxes].cpu().detach().numpy()).decode('utf')
    }


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = os.path.join(args.load_dir, args.net, args.dataset)
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    print("load checkpoint %s" % (load_name))

    input_holders = create_input_holders(args)

    cfg.CUDA = args.cuda
    # ship to cuda
    if args.cuda:
        fasterRCNN.cuda()
    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.2

    imglist = load_image_ids(args.data_split)
    num_images = len(imglist)
    num_classes = len(imdb.classes)

    print('Loaded Photo: {} images.'.format(num_images))
    imgid2idx = {}
    with open(args.outfile, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        for i, (im_file, image_id) in enumerate(imglist):
            misc_tic = time.time()

            imgid2idx[image_id] = i

            result = get_detections_from_im(fasterRCNN, im_file, image_id, input_holders, thresh)
            writer.writerow(result)

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            if (i % 100) == 0:
                print('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                    .format(i + 1, len(imglist), nms_time,\
                            nms_time * (len(imglist) - i) / 3600))

# Save features
    img_id2idx_path = '{}_imgid2idx.pkl'.format(args.outfile)
    with open(img_id2idx_path, 'wb') as f:
        cPickle.dump(imgid2idx, f)

    print('Done!')
