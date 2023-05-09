###############################################################################
#
#	load_test.py		Image detection test using saved TF model
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

#import set_working_directory
import os
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
#import tensorflow_hub as hub
from yolov3.classes import *
#from yolov3.yolov4 import Create_Yolo
from yolov3.utils import Load_Yolo_model
from yolov3.utils import detect_image

print( "Loading model:", repr( SAVED_MODEL ) )
# Recreate the exact same model, including its weights and the optimizer
yolo = tf.keras.models.load_model( SAVED_MODEL )
#yolo = tf.saved_model.load( SAVED_MODEL )
#yolo = keras.models.load_model( SAVED_MODEL )

# Show the model architecture
yolo.summary( )

test_annotations = open( "mnist_test.txt" ).readlines( )

while True:
	ID = random.randint( 0, 1999 )
	image_info = test_annotations[ ID ].split( )
	image_path = image_info[ 0 ]
	print( image_info )
	detect_image( yolo, image_path, "mnist_test.jpg", input_size = YOLO_INPUT_SIZE, show = True )#, rectangle_colors = ( 255, 0, 0 ) )
