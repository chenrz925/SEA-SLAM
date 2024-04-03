
import sys

from matplotlib.pyplot import box
import rospy
import cv2
modelpy_path = rospy.get_param('/semantic/model_scripts')
sys.path.append(modelpy_path)
sys.path.append(modelpy_path + '/YOLACT')
from YOLACT.utils.functions import MovingAverage, ProgressBar, SavePath
from YOLACT.layers.output_utils import postprocess, undo_image_transformation
from YOLACT.layers.box_utils import jaccard, mask_iou
from YOLACT.utils.augmentations import BaseTransform, FastBaseTransform
from YOLACT.data.coco import COCODetection, get_label_map
from cv_bridge.core import CvBridge
import pycocotools
from sensor_msgs.msg import Image
from pathlib import Path
from collections import defaultdict
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from YOLACT.data.config import COCO_CLASSES
from YOLACT.utils import timer
from YOLACT.yolact import Yolact
from YOLACT.data.config import COLORS
from YOLACT.data import cfg
from ncnn.model_zoo import get_model

torch.set_grad_enabled(False)


color_cache = defaultdict(lambda: {})


class inference_config:
    use_cross_class_nms = False
    use_fast_nms = True
    mask_proto_debug = False

    display_lincomb = False
    crop = True
    score_threshold = 0.45
    top_k = 15


SAVER_DIR = modelpy_path +'/YOLACT/result/'
MODEL_DIR = modelpy_path + '/YOLACT/checkpoints/'


class YOLACTServer(object):
    def __init__(self):
        self.name = 'YOLACT'
        self.model = Yolact()
        self.checkpoint_path = MODEL_DIR + \
            rospy.get_param('/semantic/checkpoint')
        self.model.load_weights(self.checkpoint_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            cudnn.fastest = True
            torch.set_default_dtype(torch.float32)
            torch.set_default_device('cuda:0')
        else:
            torch.set_default_dtype(torch.float32)
            torch.set_default_device('cpu')

        self.masked_image_pubulisher = rospy.Publisher("masked_image",
                                                       Image,
                                                       queue_size=10)
        self.model.detect.use_cross_class_nms = inference_config.use_cross_class_nms
        self.model.detect.use_fast_nms = inference_config.use_fast_nms
        cfg.mask_proto_debug = inference_config.mask_proto_debug
        print("=============Initialized YOLACT==============")
        # self.name = 'YOLACT'
        # # self.model = Yolact()
        # # self.checkpoint_path = MODEL_DIR + \
        # #     rospy.get_param('/semantic/checkpoint')
        # # self.model.load_weights(self.checkpoint_path)
        # # self.model.eval()
        # self.model = get_model(
        #     "yolact",
        #     target_size=550,
        #     confidence_threshold=0.7,
        #     nms_threshold=0.15,
        #     keep_top_k=15,
        #     num_threads=4,
        #     use_gpu=False,
        # )

        # self.masked_image_pubulisher = rospy.Publisher("masked_image",
        #                                                Image,
        #                                                queue_size=10)
        # cfg.mask_proto_debug = inference_config.mask_proto_debug
        # print("=============Initialized YOLACT==============")
    def inference(self, images):
        # prepare images
        num_images = len(images)
        if torch.cuda.is_available():
            batch = torch.from_numpy(np.concatenate(images)).cuda().float()
        else:
            batch = torch.from_numpy(np.concatenate(images)).float()

        batch = batch.reshape(
            num_images, -1, batch.shape[-2], batch.shape[-1])
        # print(images.shape)
        batch = FastBaseTransform()(batch)

        time_start = rospy.Time.now()
        with torch.no_grad():
            results = self.model(batch)
        semantic_time = (rospy.Time.now() - time_start).to_sec() * 1000
        print("predict time: %f ms \n" % semantic_time)
        timer_start = rospy.Time.now()
        r = []
        self.masked_image_ = []
        for i in range(len(images)):
            r.append(results[i]['detection'])
            mask_image,masks,boxes,classes,scores = self.prep_display(
                [results[i]], torch.from_numpy(images[i]), None, None, undo_transform=False,display_masks=False,show_opencv=False)
            self.masked_image_.append(mask_image)
            # print(mask_image.shape)
            # cv2.imshow('mask',mask_image)
            if masks is None:
                r[i]['mask']=None
                r[i]['box']=None
                r[i]['class']=None
                r[i]['score']=None
            else:
                r[i]['mask']=masks.squeeze(-1).permute(1,2,0).cpu().numpy()
                r[i]['box']=boxes
                r[i]['class']=classes
                r[i]['score']=scores
        segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
        print("Visualize time: %f ms \n" % segment_time)

        return self.masked_image_, r
    
    def inference_ncnn(self, images):
        # prepare images

        time_start = rospy.Time.now()
        boxes, masks, classes, scores = self.model(images[0])

        semantic_time = (rospy.Time.now() - time_start).to_sec() * 1000
        print("predict time: %f ms \n" % semantic_time)
        timer_start = rospy.Time.now()
        r = []
        self.masked_image_ = []
        for i in range(len(images)):
            r.append({'mask':masks,'box':boxes,'score':scores,'class':classes})
            if masks is None:
                r[i]['mask']=None
                r[i]['box']=None
                r[i]['class']=None
                r[i]['score']=None
            else:
                r[i]['mask']=[]
                r[i]['box']=[]
                r[i]['class']=[]
                r[i]['score']=[]
                for box, mask, label, score in zip(boxes, masks, classes, scores):
                    r[i]['mask'].append(mask)
                    r[i]['box'].append(box)
                    r[i]['class'].append(label)
                    r[i]['score'].append(score)
        segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
        print("Visualize time: %f ms \n" % segment_time)

        return self.masked_image_, r

    def publish_result(self, images, stamp):
        if images is None or stamp is None:
            print("empty data")
            return
        msg_img = CvBridge().cv2_to_imgmsg(images, encoding="passthrough")
        msg_img.header.stamp = stamp
        self.masked_image_pubulisher.publish(msg_img)

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str='',
                     display_masks=False, display_fps=False, display_text=False, display_bboxes=False, display_scores=False,
                     show_opencv=False):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            if torch.cuda.is_available():
                img_gpu = img.cuda() / 255.0
            else:
                img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb=inference_config.display_lincomb,
                            crop_masks=inference_config.crop,
                            score_threshold=inference_config.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:inference_config.top_k]

            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        if boxes.shape[0]==0:
            return None,None,None,None,None
        num_dets_to_consider = min(inference_config.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < inference_config.score_threshold:
                num_dets_to_consider = j
                break
        masks = masks[:num_dets_to_consider, :, :, None]
        boxes = boxes[:num_dets_to_consider,:]
        classes = classes[:num_dets_to_consider]
        scores = scores[:num_dets_to_consider]
        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed

        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (
                classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                    color = torch.Tensor(color)
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            # masks = masks[:num_dets_to_consider, :, :, None]

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).reshape(
                1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1

            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(
                    num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * \
                inv_alph_masks.prod(dim=0) + masks_color_summand

        if display_fps:
            # Draw the box for the fps on the GPU
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(
                fps_str, font_face, font_scale, font_thickness)[0]

            img_gpu[0:text_h+8, 0:text_w+8] *= 0.6  # 1 - Box alpha

        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if display_fps:
            # Draw the text on the CPU
            text_pt = (4, text_h + 2)
            text_color = [255, 255, 255]

            cv2.putText(img_numpy, fps_str, text_pt, font_face,
                        font_scale, text_color, font_thickness, cv2.LINE_AA)

        if num_dets_to_consider == 0:
            return img_numpy

        if display_text or display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (
                        _class, score) if display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(
                        text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1),
                                  (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face,
                                font_scale, text_color, font_thickness, cv2.LINE_AA)

        if show_opencv:            
            if img_numpy is not None:
                cv2.imshow("Result", img_numpy)
                cv2.waitKey(1)
        
        return img_numpy,masks,boxes,(classes+1),scores
