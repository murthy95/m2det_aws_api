import os
import cv2
import numpy as np
import time
import sys
import io
import json 
from io import BytesIO

from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from utils.core import *
from utils.pycocotools.coco import COCO

from PIL import Image
import flask

global cfg

cfg = Config.fromfile('configs/m2det512_vgg.py')

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(cfg.model.m2det_config.num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(cfg.model.m2det_config.num_classes)]
cats = [_.strip().split(',')[-1] for _ in open('data/coco_labels.txt','r').readlines()]
labels = tuple(['__background__'] + cats)

def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)

    return imgcv

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        trained_model = '/opt/ml/model/m2det512_vgg.pth'
        #trained_model = '../../m2det512_vgg.pth'
        
        anchor_config = anchors(cfg)
        print_info('The Anchor info: \n{}'.format(anchor_config))
        priorbox = PriorBox(anchor_config)
        net = build_net('test',
                        size = cfg.model.input_size,
                        config = cfg.model.m2det_config)
        init_net(net, cfg, trained_model)
        print_info('===> Finished constructing and loading model',['yellow','bold'])
        net.eval()
        with torch.no_grad():
            priors = priorbox.forward()
            if cfg.test_cfg.cuda:
                net = net.cuda()
                priors = priors.cuda()
                cudnn.benchmark = True
            else:
                net = net.cpu()
        _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
        detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
        
        return net, priors, _preprocess, detector 
			

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        net, priors, _preprocess, detector = cls.get_model()
        np_image = np.array(input)
        image = np_image[:, :, ::-1].copy() 
        
        loop_start = time.time()
        w,h = image.shape[1],image.shape[0]
        img = _preprocess(image).unsqueeze(0)
        if cfg.test_cfg.cuda:
            img = img.cuda()
        scale = torch.Tensor([w,h,w,h])
        out = net(img)
        boxes, scores = detector.forward(out, priors)
        boxes = (boxes[0]*scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        for j in range(1, cfg.model.m2det_config.num_classes):
            inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            soft_nms = cfg.test_cfg.soft_nms
            keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist()+[j] for _ in c_dets])
        
        loop_time = time.time() - loop_start
        allboxes = np.array(allboxes)
        boxes = allboxes[:,:4]
        scores = allboxes[:,4]
        cls_inds = allboxes[:,5]
        
#        response_str = ''
#        response_str = response_str+'\n'.join(['pos:{}, ids:{}, score:{:.3f}'.format('(%.1f,%.1f,%.1f,%.1f)' % (o[0],o[1],o[2],o[3]) \
#               ,labels[int(oo)],ooo) for o,oo,ooo in zip(boxes,cls_inds,scores)])
#        #print (response_str)
#        return response_str
#        
        response = {}
        response['pos'] = list(boxes.reshape(-1))
        response['cls_inds'] = list(cls_inds)
        response['scores'] = list(scores)

        return response
        
        #fps = -1
        #im2show = draw_detection(image, boxes, scores, cls_inds, fps)
        ##print bbox_pred.shape, iou_pred.shape, prob_pred.shape
        #
        ##cv2.imwrite('{}_m2det.jpg'.format(image_path.split('.')[0]), im2show)
        ##print('Saved results successfully !')
        #img = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
        #im_pil = Image.fromarray(img)
        #return im_pil

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    data = None
    if not flask.request.content_type.startswith('image'):
        return flask.Response(response='\n', status=404, mimetype='application/json')
    
    img = Image.open(BytesIO(flask.request.data)).convert('RGB')
    predictions = ScoringService.predict(img)
#    result = {'result': predictions}
    return flask.Response(response=json.dumps(predictions), status=200, mimetype='application/json')

if __name__ == '__main__':
	app.debug = True
	app.run(host='0.0.0.0', port=8080)

