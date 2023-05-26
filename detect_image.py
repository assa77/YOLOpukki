###############################################################################
#
#	detect_image.py		Image detection test using YOLO
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

#import set_working_directory
import os
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
import random
import cv2
#import tensorflow as tf
from yolov3.classes import *
from yolov3.utils import Load_Yolo_model
from yolov3.utils import detect_image

yolo = Load_Yolo_model( )

test_annotations = open( TEST_ANNOTATIONS ).readlines( )

while True:
	ID = random.randint( 0, 1999 )
	image_info = test_annotations[ ID ].split( )
	image_path = image_info[ 0 ]
	print( image_info )
	if detect_image( yolo, image_path, "test.jpg", input_size = YOLO_INPUT_SIZE, show = True ) is None:#, rectangle_colors = ( 255,0,0 ) ) is None:
		break

cv2.destroyAllWindows( )
