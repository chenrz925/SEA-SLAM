import os
from numpy import imag
import sys
import rospy
modelpy_path = rospy.get_param('/semantic/model_scripts')
sys.path.append(modelpy_path)
from MaskRCNN.mrcnn.model import MaskRCNN
from MaskRCNN.mrcnn import visualize
from MaskRCNN.samples.coco import coco
from cv_bridge.core import CvBridge
from sensor_msgs.msg import Image

SAVER_DIR = modelpy_path+'MaskRCNN/result/'
MODEL_DIR = modelpy_path + 'MaskRCNN/checkpoints/'

class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128


class MaskRCNNServer(object):
    def __init__(self):
        config = InferenceConfig()
        config.display()

        self.name = 'MaskRCNN'
        self.model = MaskRCNN(mode="inference",
                              model_dir=MODEL_DIR,
                              config=config)
        self.checkpoint_path = MODEL_DIR + \
            rospy.get_param('/semantic/checkpoint')
        self.model.load_weights(self.checkpoint_path, by_name=True)
        self.masked_image_pubulisher = rospy.Publisher("masked_image",
                                                 Image,
                                                 queue_size=10)
        print("=============Initialized MaskRCNN==============")

    def inference(self, images):
        time_start = rospy.Time.now()
        results = self.model.detect(images, verbose=1)
        semantic_time = (rospy.Time.now() - time_start).to_sec() * 1000
        print("predict time: %f ms \n" % semantic_time)
        timer_start = rospy.Time.now()
        r = results[0]
        self.masked_image_ = visualize.ros_semantic_result(images[0],
                                                           r['box'],
                                                           r['mask'],
                                                           r['class'],
                                                           class_names,
                                                           r['score'],
                                                           show_opencv=False)

        segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
        print("Visualize time: %f ms \n" % segment_time)
        # print(self.masked_image_.shape)
        return self.masked_image_, results

    def publish_result(self, images, stamp):
        if images is None or stamp is None:
            print("empty data")
            return 
        msg_img = CvBridge().cv2_to_imgmsg(images,encoding="passthrough")
        msg_img.header.stamp = stamp
        self.masked_image_pubulisher.publish(msg_img)
