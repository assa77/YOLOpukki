###############################################################################
#
#	detect_mnist.py		Image detection test using YOLO
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

#import set_working_directory
import os
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
import cv2
#import numpy as np
import random
#import time
#import tensorflow as tf
#import tensorflow_hub as hub
from yolov3.classes import *
#from yolov3.yolov4 import Create_Yolo
from yolov3.utils import Load_Yolo_model
from yolov3.utils import detect_image
#from yolov3.utils import detect_video
#from yolov3.utils import detect_realtime
#from yolov3.utils import detect_video_realtime_mp

yolo = Load_Yolo_model( )

#detect_image( yolo, image_path, "mnist_test.jpg", input_size = YOLO_INPUT_SIZE, show = True )#, rectangle_colors = ( 255, 0, 0 ) )
#detect_video( yolo, video_path, "", input_size = YOLO_INPUT_SIZE, show = False )#, rectangle_colors = ( 255, 0, 0 ) )
#detect_realtime( yolo, None, "mnist_out.mp4", input_size = YOLO_INPUT_SIZE, show = True )#, rectangle_colors = ( 255, 0, 0 ) )

#detect_video_realtime_mp( None, "", "mnist_out.mp4", input_size = YOLO_INPUT_SIZE, show = True, realtime = True )#, rectangle_colors = ( 255, 0, 0 ) )
#detect_video_realtime_mp( None, video_path, "mnist_out.mp4", input_size = YOLO_INPUT_SIZE, show = False, realtime = False )#, rectangle_colors = ( 255, 0, 0 ) )

test_annotations = open( "mnist_test.txt" ).readlines( )

while True:
	ID = random.randint( 0, 1999 )
	image_info = test_annotations[ ID ].split( )
	image_path = image_info[ 0 ]
	print( image_info )
	if detect_image( yolo, image_path, "mnist_test.jpg", input_size = YOLO_INPUT_SIZE, show = True ) is None:#, rectangle_colors = ( 255,0,0 ) ) is None:
		break

cv2.destroyAllWindows( )
