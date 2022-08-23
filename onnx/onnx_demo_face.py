import numpy as np
import sys, os
from math import sqrt
from math import exp
import cv2
import onnxruntime as ort


CLASSES = ('background', 'face')

class_num = len(CLASSES)
num_priors = 3

image_h = 240
image_w = 320

min_sizes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]

feature_maps = [[30, 40], [15, 20], [8, 10], [4, 5]]

steps = [8, 16, 32, 64]

variances = [0.1, 0.2]
offset = 0.5

head_num = 4

nmsThre = 0.45
objThre = 0.5

priorbox_mean = []


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def priorBox():
    for index in range(head_num):
        h = feature_maps[index][0]
        w = feature_maps[index][1]

        scale_w = image_w / steps[index]
        scale_h = image_h / steps[index]

        for i in range(h):
            for j in range(w):
                
                for min_box in min_sizes[index]:
                
                    cx = (j + offset) / scale_w
                    cy = (i + offset) / scale_h

                    cw = min_box / image_w
                    ch = min_box / image_h
                    
                    cx = cx if cx <= 1 else 1
                    cy = cy if cy <= 1 else 1
                    cw = cw if cw <= 1 else 1
                    ch = ch if ch <= 1 else 1
                    
                    priorbox_mean.append(cx)
                    priorbox_mean.append(cy)
                    
                    priorbox_mean.append(cw)
                    priorbox_mean.append(ch)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThre:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def postprocess(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []
    
    output = []
    for i in range(len(out)):
        print(len(out[i].reshape((-1))))
        output.append(out[i].reshape((-1)))

    priorbox_index = -4

    for head in range(head_num):
        
        conf = output[head * 2]
        loc = output[head * 2 + 1]

        for h in range(feature_maps[head][0]):
            for w in range(feature_maps[head][1]):
                if head == 0 or head == 3:
                    num_priors = 3
                else:
                    num_priors = 2

                for pri in range(num_priors):
                    priorbox_index += 4

                    conf_temp = []
                    softmaxSum = 0

                    for cl in range(class_num):
                        conf_t = conf[feature_maps[head][0] * feature_maps[head][1] * (pri * class_num + cl) + h * feature_maps[head][1] + w]
                        conf_t_exp = exp(conf_t)
                        softmaxSum += conf_t_exp
                        conf_temp.append(conf_t_exp)

                    loc_temp = []
                    for lc in range(4):
                        loc_t =  loc[feature_maps[head][0] * feature_maps[head][1] * (pri * 4 + lc) + h * feature_maps[head][1] + w]
                        loc_temp.append(loc_t)

                    for clss in range(1, class_num, 1):
                        conf_temp[clss] /= softmaxSum

                        if conf_temp[clss] > objThre:
                            bx = priorbox_mean[priorbox_index + 0] + (loc_temp[0] * variances[0] * priorbox_mean[priorbox_index + 2])
                            by = priorbox_mean[priorbox_index + 1] + (loc_temp[1] * variances[0] * priorbox_mean[priorbox_index + 3])
                            bw = priorbox_mean[priorbox_index + 2] * exp(loc_temp[2] * variances[1])
                            bh = priorbox_mean[priorbox_index + 3] * exp(loc_temp[3] * variances[1])

                            xmin = (bx - bw / 2) * img_w
                            ymin = (by - bh / 2) * img_h
                            xmax = (bx + bw / 2) * img_w
                            ymax = (by + bh / 2) * img_h

                            if xmin >= 0 and ymin >= 0 and xmax <= img_w and ymax <= img_h:
                                box = DetectBox(clss, conf_temp[clss], xmin, ymin, xmax, ymax)
                                detectResult.append(box)

    # NMS 过程
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)
    return predBox


def preprocess(src):
    img = cv2.resize(src, (image_w, image_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img - 127.5
    img = img * 0.007843    
    return img


def detect(imgfile, model_path):
    origimg = cv2.imread(imgfile)
    img_h, img_w = origimg.shape[:2]
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    ort_session = ort.InferenceSession(model_path)
    res = (ort_session.run(None, {'input': img}))

    out = []
    for i in range(len(res)):
        out.append(res[i])

    predbox = postprocess(out, img_h, img_w)

    print(len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(origimg, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + "%.2f" % score
        cv2.putText(origimg, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite('./test_result.jpg', origimg)
    # cv2.imshow("test", origimg)
    # cv2.waitKey(0)


if __name__ == '__main__':
    print('This is main .... ')
    priorBox()
    #print(priorbox_mean[0:100])
    print('priorbox:', len(priorbox_mean))
    detect('./test.jpg', './RFB_320.onnx')
