import requests
import sys
import json
import numpy as np
import cv2
from configs.CC import Config
import time
#import boto3


#endpoint='m2det-endpoint'
#runtime_client = boto3.client('sagemaker-runtime')
#response = runtime_client.invoke_endpoint(EndpointName=endpoint, 
#           Body=img_encoded.tostring(), ContentType='text/csv')
##files= {'image': open(sys.argv[1], 'rb')}
##r = requests.post(url, files=files)
#print(response['Body'].read().decode('ascii')) 

global cfg

cfg = Config.fromfile('configs/m2det512_vgg.py')

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
print(cfg.model.m2det_config.num_classes)
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

url = "aws_api_gateway_url_here"
image = sys.argv[1]
start = time.time()
content_type = 'image/png'
headers = {'Content-Type': content_type}
response = requests.post(url, data=open(image, 'rb'), headers=headers)
response = json.loads(response.content.decode())
end1 = time.time()
print(response)
print (end1 - start)

image = cv2.imread(image)
boxes = np.array(response['pos']).reshape((-1, 4))
scores = response['scores']
cls_inds = response['cls_inds']
fps = -1
im2show = draw_detection(image, boxes, scores, cls_inds, fps)
end2 = time.time()
print (end2-start)
image_path = 'cats'
cv2.imwrite('{}_m2det.jpg'.format(image_path.split('.')[0]), im2show)

