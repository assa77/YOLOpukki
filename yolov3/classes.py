###############################################################################
#
#	classes.py		Main YOLO configuration module
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

from yolov3.configs import *

def read_class_names( class_file_name ):
	# loads class name from a file
	names = { }
	with open( class_file_name, 'r' ) as data:
		for ID, name in enumerate( data ):
			names[ ID ] = name.strip( '\n' )
	return names

if YOLO_TYPE == "yolov4":
	YOLO_ANCHORS = [
				[ [ 12,  16 ],  [ 19,  36 ],  [ 40,  28 ] ],
				[ [ 36,  75 ],  [ 76,  55 ],  [ 72,  146 ] ],
				[ [ 142, 110 ], [ 192, 243 ], [ 459, 401 ] ]
				]
if YOLO_TYPE == "yolov3":
	YOLO_ANCHORS = [
				[ [ 10,  13 ],  [ 16,  30 ],  [ 33,  23 ] ],
				[ [ 30,  61 ],  [ 62,  45 ],  [ 59,  119 ] ],
				[ [ 116, 90 ],  [ 156, 198 ], [ 373, 326 ] ]
				]

if TRAIN_YOLO_TINY:
	YOLO_STRIDES = [ 16, 32 ]
	YOLO_ANCHORS = [
	# Uncomment next line to use the default COCO weights
	#			[ [ 23,  27 ],  [ 37,  58 ],  [ 81,  82 ] ],
				[ [ 10,  14 ],  [ 23,  27 ],  [ 37,  58 ] ],
				[ [ 81,  82 ],  [ 135, 169 ], [ 344, 319 ] ]
				]
else:
	YOLO_STRIDES = [ 8, 16, 32 ]

if not YOLO_CUSTOM_WEIGHTS:
	if YOLO_TYPE == "yolov4":
		DARKNET_WEIGHTS = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
	if YOLO_TYPE == "yolov3":
		DARKNET_WEIGHTS = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
	CLASSES = YOLO_CLASSES
	print( "Using darknet weights:", repr( DARKNET_WEIGHTS ) )
else:
	DARKNET_WEIGHTS = f"./{TRAIN_CHECKPOINTS_FOLDER}/{TRAIN_MODEL_NAME}"
	if TRAIN_YOLO_TINY:
		DARKNET_WEIGHTS += "_tiny"
	print( "Using custom weights:", repr( DARKNET_WEIGHTS ) )
	CLASSES = TRAIN_CLASSES
if TRAIN_YOLO_TINY:
	SAVED_MODEL = f"./{TRAIN_CHECKPOINTS_FOLDER}/{YOLO_TYPE}-tiny-{YOLO_INPUT_SIZE}"
else:
	SAVED_MODEL = f"./{TRAIN_CHECKPOINTS_FOLDER}/{YOLO_TYPE}-{YOLO_INPUT_SIZE}"
if YOLO_FRAMEWORK == "tf":	# using TensorFlow framework
	print( "Using saved model:", repr( SAVED_MODEL ) )
elif YOLO_FRAMEWORK == "trt":	# using TensorRT framework
#	import tensorflow.contrib.tensorrt as trt
#	from tensorflow.python.compiler.tensorrt import trt_convert as trt
#	import tensorrt as trt
	if TRAIN_YOLO_TINY:
		SAVED_TRT_MODEL = f"./{TRAIN_CHECKPOINTS_FOLDER}/{YOLO_TYPE}-tiny-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}"
	else:
		SAVED_TRT_MODEL = f"./{TRAIN_CHECKPOINTS_FOLDER}/{YOLO_TYPE}-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}"
	print( "Using saved model:", repr( SAVED_TRT_MODEL ) )

CLASS_NAMES = read_class_names( CLASSES )
NUM_CLASSES = len( CLASS_NAMES )
print( f"Using {NUM_CLASSES} class(es) from:", repr( CLASSES ) )
