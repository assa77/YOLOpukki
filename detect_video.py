###############################################################################
#
#	detect_video.py		Real-time video detection test using YOLO
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

#import set_working_directory
import os
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
#import cv2
#import numpy as np
#import random
#import time
#import tensorflow as tf
#import tensorflow_hub as hub
from yolov3.classes import *
from generator_zla import generator
#from yolov3.yolov4 import Create_Yolo
from yolov3.utils import Load_Yolo_model
from yolov3.utils import detect_realtime
#from yolov3.utils import detect_video_realtime_mp

def main( ):
	yolo = Load_Yolo_model( )
	detect_realtime( yolo, generator, "mnist_out.mp4", input_size = YOLO_INPUT_SIZE, width = 640, height = 480, show = True )#, rectangle_colors = ( 255, 0, 0 ) )

#	detect_video_realtime_mp( generator, "", "mnist_out.mp4", input_size = YOLO_INPUT_SIZE, width = 640, height = 480, show = True, realtime = True )#, rectangle_colors = ( 255, 0, 0 ) )

if __name__ == '__main__':
	main( )
