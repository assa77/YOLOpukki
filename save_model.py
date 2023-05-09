###############################################################################
#
#	save_model.py		Saves model in TensorFlow formats
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

import os
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
import sys
import tensorflow as tf
from yolov3.classes import *
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights

yolo = Create_Yolo( input_size = YOLO_INPUT_SIZE )
if not YOLO_CUSTOM_WEIGHTS:
	load_yolo_weights( yolo, DARKNET_WEIGHTS )	# use Darknet weights
else:
	yolo.load_weights( DARKNET_WEIGHTS )		# use custom weights

yolo.summary( )

model_name = SAVED_MODEL
yolo.save( model_name )
print( "Model saved to:", repr( model_name ) )
model_name += '.h5'
yolo.save( model_name )
#tf.keras.models.save_model( yolo, model_name )
print( "Model saved to:", repr( model_name ) )
