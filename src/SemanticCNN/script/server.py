#!/usr/bin/env python

import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import box
from matplotlib.transforms import Bbox
import actionlib
import threading
from sensor_msgs.msg import Image
import rospy
import sys
import os
from semantic_cnn.msg import *
import semantic_cnn.msg
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

modelpy_path = rospy.get_param('/semantic/model_scripts')
sys.path.append(modelpy_path)

model_name = rospy.get_param('/semantic/model_name')
model_server = None
if model_name == 'maskrcnn':
    sys.path.append(f'{modelpy_path}/MaskRCNN')
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()    
    from MaskRCNN.MaskRCNN_server import MaskRCNNServer
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction=0.8
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("using MaskRCNN")
    model_server = MaskRCNNServer
elif model_name == 'yolact':
    sys.path.append(f'{modelpy_path}/YOLACT')
    from YOLACT.YOLACT_server import YOLACTServer
    print("using YOLACT")
    model_server = YOLACTServer
elif model_name == 'sparseinst':
    from SparseInst.SparseInst_server import SparseInstServer
    print("using SparseInst")
    model_server = SparseInstServer


# =========global params===============
# Totoal time consuming
total_time = float(0.0)
# Total number of images
total_number = int(0)
# process image num
batch_size = rospy.get_param('/semantic/batch_size')
# To sync semantic segmentation thread and action server thread
cn_task = threading.Condition()
cn_ready = threading.Condition()

result = None
class_needed = [1,2,3,4,5,6,7,8,9,57]
# ==========         server     ===============


class SemanticServer(object):
    _feedback = semantic_cnn.msg.semanticFeedback()
    _result = semantic_cnn.msg.semanticResult()

    def __init__(self):
        # set action server
        self._action_name = rospy.get_param(
            'semantic/action_name', '/semantic_server')
        print('action server: ', self._action_name)
        self._action_server = actionlib.SimpleActionServer(
            self._action_name,
            semantic_cnn.msg.semanticAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self._action_server.start()
        print("Semantic action server start...")
        print('Is CUDA available: ', torch.cuda.is_available())
        print('CUDA device: ', torch.cuda.current_device())

    def execute_cb(self, goal):
        global total_time
        global total_number
        global color_image
        global stamp
        global masked_image
        global result

        color_image = []
        stamp = []
        if not self._action_server.is_active():
            print("[Error] semantic action server cannot active")
            return
        time_task_start = rospy.Time.now()
        print("----------------------------------")
        print("ID: %d" % goal.id)

        # prepare images
        for i in range(batch_size):
            try:
                color_image.append(
                    CvBridge().imgmsg_to_cv2(goal.image[i], 'bgr8')
                )
                stamp.append(goal.image[i].header.stamp)
            except CvBridgeError as e:
                print(e)
                return

        time_start = rospy.Time.now()
        if np.any(color_image[0]):
            with cn_task:
                print("Inform new task")
                cn_task.notifyAll()
            with cn_ready:
                cn_ready.wait()
                print("semantic result ready")

        # average time consuming
        semantic_time = (rospy.Time.now()-time_start).to_sec()*1000
        print("semantic time: %f ms \n" % semantic_time)

        total_time = float(total_time)+float(semantic_time)
        total_number = total_number+1
        if int(total_number) > 0:
            average_time = total_time/float(total_number)
            print("Average time: %f" % average_time)

        # init msg
        object_num = []
        label_msg = []
        score_msg = []
        object_msg = []
        bbox_msg = []
        class_msg = []
        if model_name == 'sparseinst':
            boxes = result[i]['box']
            masks = result[i]['mask']
            class_ids = result[i]['class']
            # scores = result[i]['score']
            if masks is None:
                object_num.append(0)
                label_img = np.zeros(color_image[0].shape[:2], dtype=np.uint8)
            
            else:
                objnum=0
                N = masks.shape[0]
                label_img = np.zeros(masks.shape[1:], dtype=np.uint8)
                object_img = np.zeros(masks.shape[1:], dtype=np.uint8)
                for i in range(N):
                    # if(class_ids[i]<1 or class_ids[i]>9):
                    if(class_ids[i] != 1):
                        continue
                    objnum+=1
                    # mask = masks[]
                    # merge labels into one single image
                    cur_mask = (masks[i,:,:]!=0)
                    class_msg.append(class_ids[i])
                    label_img = (cur_mask * class_ids[i]).astype(np.uint8)
                    object_img = (cur_mask * objnum).astype(np.uint8)
                    object_msg.append(CvBridge().cv2_to_imgmsg(
                        object_img.astype(np.uint8), encoding='mono8'))
                    label_msg.append(CvBridge().cv2_to_imgmsg(
                        label_img.astype(np.uint8), encoding='mono8'))
                # print('shape of label image : ' , label_img.shape)
                # set msg
                object_num.append(objnum)
        else:
            for i in range(batch_size):
                boxes = result[i]['box']
                masks = result[i]['mask']
                class_ids = result[i]['class']
                # scores = result[i]['score']
                if boxes is None:
                    object_num.append(0)
                    label_img = np.zeros(color_image[0].shape[:2], dtype=np.uint8)
                
                else:
                    objnum=0
                    N = len(boxes)
                    label_img = np.zeros((480,640), dtype=np.uint8)
                    object_img = np.zeros((480,640), dtype=np.uint8)
                    for i in range(N):
                        # if(class_ids[i]<1 or class_ids[i]>9):
                        if(class_ids[i] != 1):
                            continue
                        objnum+=1
                        # mask = masks[]
                        # merge labels into one single image
                        cur_mask = (masks[:,:,i]!=False)
                        class_msg.append(class_ids[i])
                        label_img = (cur_mask * class_ids[i]).astype(np.uint8)
                        object_img = (cur_mask * objnum).astype(np.uint8)
                        object_msg.append(CvBridge().cv2_to_imgmsg(
                            object_img.astype(np.uint8), encoding='mono8'))
                        label_msg.append(CvBridge().cv2_to_imgmsg(
                            label_img.astype(np.uint8), encoding='mono8'))
                        for x in boxes[i]:
                            bbox_msg.append(x)
                    # print('shape of label image : ' , label_img.shape)
                    # set msg
                    object_num.append(objnum)
                    

        # result
        self._result.id = goal.id
        self._result.label = label_msg
        self._result.label_object = object_msg
        self._result.bbox = bbox_msg
        self._result.object_num = object_num
        self._result.class_id = class_msg

        # feedback
        self._feedback.complete = True
        self._action_server.set_succeeded(self._result)
        self._action_server.publish_feedback(self._feedback)

        # calculate time consuming
        # print("Detect Object: %f  \n" % object_num[0],object_num[1])
        time_task_start = (rospy.Time.now()-time_task_start).to_sec()*1000
        print("Time of each request: %f ms \n" % time_task_start)


def worker(model_server, is_publish):
    global color_image
    global stamp
    global result

    color_image = []
    stamp=[]
    model_server = model_server()
    while True:
        with cn_task:
            cn_task.wait()
            print("New task comming")
            timer_start = rospy.Time.now()
            masked_image, result = model_server.inference(color_image)
            segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
            print("%s segment time for curent image: %f ms \n" %(model_server.name, segment_time))
            with cn_ready:
                cn_ready.notifyAll()
                if is_publish:
                    model_server.publish_result(masked_image,stamp)

if __name__ == '__main__':
    rospy.init_node("semantic_server", anonymous=False)

    thread_semantic = threading.Thread(
        name=model_name,
        target=worker,
        args=(
            model_server,
            False
    ))
    thread_semantic.start()

    action_server = SemanticServer()
    print('Setting up %s Action Server...' % model_name)
    print("waiting for worker threads")
    main_thread = threading.currentThread()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('shutting down')

