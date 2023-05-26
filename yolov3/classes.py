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

if YOLO_TINY:
	YOLO_STRIDES = [ 16, 32 ]
	YOLO_ANCHORS = [
	# Uncomment next line to use the default COCO weights
	#			[ [ 23,  27 ],  [ 37,  58 ],  [ 81,  82 ] ],
				[ [ 10,  14 ],  [ 23,  27 ],  [ 37,  58 ] ],
				[ [ 81,  82 ],  [ 135, 169 ], [ 344, 319 ] ]
				]
else:
	YOLO_STRIDES = [ 8, 16, 32 ]

TINY_SUFFIX = "-tiny"
if YOLO_TYPE == "yolov4":
	DARKNET_WEIGHTS = f"./{YOLO_CHECKPOINTS}/{YOLO_V4_TINY_WEIGHTS if YOLO_TINY else YOLO_V4_WEIGHTS}"
if YOLO_TYPE == "yolov3":
	DARKNET_WEIGHTS = f"./{YOLO_CHECKPOINTS}/{YOLO_V3_TINY_WEIGHTS if YOLO_TINY else YOLO_V3_WEIGHTS}"
CUSTOM_WEIGHTS = f"./{YOLO_CHECKPOINTS}/{TRAIN_MODEL_NAME}{TINY_SUFFIX if YOLO_TINY else ''}"
if not YOLO_CUSTOM_WEIGHTS:
	CLASSES = YOLO_CLASSES
	TRAIN_ANNOTATIONS = YOLO_TRAIN
	TEST_ANNOTATIONS = YOLO_VAL
	print( "Using darknet weights:", repr( DARKNET_WEIGHTS ) )
	SAVED_MODEL = f"./{YOLO_CHECKPOINTS}/{YOLO_TYPE}{TINY_SUFFIX if YOLO_TINY else ''}-{YOLO_INPUT_SIZE}"
else:
	CLASSES = TRAIN_CLASSES
	print( "Using custom weights:", repr( CUSTOM_WEIGHTS ) )
	SAVED_MODEL = f"{CUSTOM_WEIGHTS}-{YOLO_INPUT_SIZE}"
if YOLO_FRAMEWORK == "tf":	# using TensorFlow framework
	print( "Using saved model:", repr( SAVED_MODEL ) )
elif YOLO_FRAMEWORK == "trt":	# using TensorRT framework
#	import tensorflow.contrib.tensorrt as trt
#	from tensorflow.python.compiler.tensorrt import trt_convert as trt
#	import tensorrt as trt
	if not YOLO_CUSTOM_WEIGHTS:
		SAVED_TRT_MODEL = f"./{YOLO_CHECKPOINTS}/{YOLO_TYPE}{TINY_SUFFIX if YOLO_TINY else ''}-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}"
	else:
		SAVED_TRT_MODEL = f"{CUSTOM_WEIGHTS}-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}"
	print( "Using saved model:", repr( SAVED_TRT_MODEL ) )

CLASS_NAMES = read_class_names( CLASSES )
NUM_CLASSES = len( CLASS_NAMES )
print( f"Using {NUM_CLASSES} class(es) from:", repr( CLASSES ) )
print( repr( CLASS_NAMES ) )
