import cv2
import json
import numpy as np
import os
import time
import glob
import argparse
import numpy as np
import sys
import qcsnpe as qc

CPU = 0
GPU = 1
DSP = 2

def postprocess_boxes(boxes, scale, height, width):
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes

def draw_boxes(image, boxes, scores, labels, colors, classes):
    for b, l, s in zip(boxes, labels, scores):
        class_id = int(l)
        class_name = classes[class_id]
    
        xmin, ymin, xmax, ymax = list(map(int, b))
        score = '{:.4f}'.format(s)
        color = colors[class_id]
        label = '-'.join([class_name, score])
    
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def preprocess_image(image, image_size):
    # image, RGB
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant') 

    return image, scale

def _nms_boxes(boxes, scores, iou_thresh=0.45):
	x = boxes[:, 0]
	y = boxes[:, 1]
	w = boxes[:, 2] -  boxes[:, 0]
	h = boxes[:, 3] -  boxes[:, 1]
	areas = (w)* (h)
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)

		xx1 = np.maximum(x[i], x[order[1:]])
		yy1 = np.maximum(y[i], y[order[1:]])
		xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
		yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

		w1 = np.maximum(0.0, xx2 - xx1 + 1)
		h1 = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w1 * h1

		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(ovr <= iou_thresh)[0]
		order = order[inds + 1]

	keep = np.array(keep)

	return keep


def filter_detections(
                boxes,
                classification,
                nms=True,
                class_specific_filter=True,
                score_threshold=0.4,
                max_detections=100,
                nms_threshold=0.5,
                parallel_iterations=32,
                detect_quadrangle=False,
            ):
    box_class_scores = np.max(classification, axis=1)
    box_classes = np.argmax(classification, axis=1)
    pos = np.where(box_class_scores >= score_threshold)

    fil_boxes = boxes[pos]
    fil_classes = box_classes[pos]
    fil_scores = box_class_scores[pos]
    nboxes, nclasses, nscores = [], [], []
    for c in set(fil_classes):
        inds = np.where(fil_classes == c)
        b = fil_boxes[inds]
        c = fil_classes[inds]
        s = fil_scores[inds]
        keep = _nms_boxes(b, s, nms_threshold)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_folder", default=None, help="image_folder")
    ap.add_argument("-v", "--vid", default=None,help="cam/video_path")
    args = vars(ap.parse_args())

    im_folder_path =  args["img_folder"]
    vid = args["vid"]

    if vid == None and im_folder_path == None:
        print("required command line args atleast ----img_folder <image folder path> or --vid <cam/video_path>")
        exit(0)

    phi = 1
    weighted_bifpn = True
    model_path = 'model_data/EfficientDet.dlc'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    classes = {value['id'] - 1: value['name'] for value in json.load(open('model_data/coco_90.json', 'r')).values()}
    num_classes = 90
    score_threshold = 0.4
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    out_layers = ["concat_34","concat_35"]
    model = qc.qcsnpe(model_path,out_layers ,CPU)

    # image inference
    if im_folder_path is not None:
        for image_path in glob.glob(im_folder_path + '/*.jpg'):
            image = cv2.imread("path_to_image_folder")
            src_image = image.copy()
            # BGR -> RGB
            image = image[:, :, ::-1]
            h, w = image.shape[:2]

            image, scale = preprocess_image(image, image_size=image_size)
            # run network
            start = time.time()

            out = model.predict(image)
            classification = out["StatefulPartitionedCall:0"]
            boxes = out["StatefulPartitionedCall:1"]
            classification = np.array(classification)
            boxes = np.array(boxes)
            classification = np.reshape(classification, (1, 76725, 90))
            boxes = np.reshape(boxes, (1, 76725, 4))
            boxes, labels, scores = filter_detections(
                    boxes[0],classification[0],nms=True,class_specific_filter=True,score_threshold=0.4,max_detections=100,
                    nms_threshold=0.5,parallel_iterations=32,detect_quadrangle=False
                    )

            print(type(boxes))
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

            # # select indices which have a score above the threshold
            indices = np.where(scores[:] > score_threshold)[0]
            # select those detections
            boxes = boxes[indices]
            labels = labels[indices]

            draw_boxes(src_image, boxes, scores, labels, colors, classes)
            cv2.imwrite("Frame.jpg" , src_image)

            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.imshow('image', src_image)
            #cv2.waitKey(0)
            
    # video inference
    if vid is not None:
        if vid == "cam":
            video_capture = cv2.VideoCapture(0)
        else:
            video_capture = cv2.VideoCapture(vid)

        while True:
            ret, image = video_capture.read()  # frame shape 640*480*3
            if ret:
                src_image = image.copy()
                # BGR -> RGB
                image = image[:, :, ::-1]
                h, w = image.shape[:2]

                img, scale = preprocess_image(image, image_size=image_size)
                image = cv2.resize(image, (640,640))
                # run network
                start = time.time()
                out = model.predict(img)
                classification = out["StatefulPartitionedCall:0"]
                boxes = out["StatefulPartitionedCall:1"]
                classification = np.array(classification)
                boxes = np.array(boxes)
                
                classification = np.reshape(classification, (1, 76725, 90))
                boxes = np.reshape(boxes, (1, 76725, 4))
                boxes, labels, scores = filter_detections(
                        boxes[0],classification[0],nms=True,class_specific_filter=True,score_threshold=0.4,max_detections=100,
                        nms_threshold=0.5,parallel_iterations=32,detect_quadrangle=False
                        )

                boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

                # # select indices which have a score above the threshold
                indices = np.where(scores[:] > score_threshold)[0]

                # select those detections
                boxes = boxes[indices]
                labels = labels[indices]

                draw_boxes(src_image, boxes, scores, labels, colors, classes)

                #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                #cv2.imshow('image', src_image)
                cv2.imwrite("image.jpg", src_image)
                #k = cv2.waitKey(1)
                #if k == ord('q'):
                    #break

if __name__ == '__main__':
    main()

