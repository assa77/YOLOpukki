###############################################################################
#
#	convert_to_trt_v2.py	Converts saved TF model to TensorRT (using V2
#				converter)
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

import os
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
import sys
import copy
import tensorflow as tf
import numpy as np
#from yolov3.dataset import Dataset

physical_devices = tf.config.experimental.list_physical_devices( 'GPU' )
if len( physical_devices ) > 0:
	try: tf.config.experimental.set_memory_growth( physical_devices[ 0 ], True )
	except RuntimeError: pass
from yolov3.configs import *
#from tensorflow.experimental import tensorrt as trt
from tensorflow.python.compiler.tensorrt import trt_convert as trt

MAX_BATCH_SIZE = 128	#100

def calibration_input_fn( ):
#	testset = Dataset( 'test', MAX_BATCH_SIZE )
#	for image_data, _ in testset:
#		yield ( image_data, )
	for i in range( MAX_BATCH_SIZE ):
		batched_input = np.random.random( ( 1, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3 ) ).astype( np.float32 )
		batched_input = tf.constant( batched_input )
		yield ( batched_input, )

def convert_to_trt( quantize_mode ):
	conversion_params = copy.deepcopy( trt.DEFAULT_TRT_CONVERSION_PARAMS )

	match YOLO_TRT_QUANTIZE_MODE:
		case 'FP16':
			precision = trt.TrtPrecisionMode.FP16
		case 'FP32':
			precision = trt.TrtPrecisionMode.FP32
		case _:
			precision = trt.TrtPrecisionMode.INT8
			conversion_params = conversion_params._replace( use_calibration = True )

	conversion_params = conversion_params._replace(
		precision_mode = precision,
		max_workspace_size_bytes = 2 << 32,
		maximum_cached_engines = 16	#100
	#	minimum_segment_size = 3,
	#	allow_build_at_runtime = True,
	#	use_calibration = True
		 )

	converter = trt.TrtGraphConverterV2(
		input_saved_model_dir = SAVED_MODEL,
		conversion_params = conversion_params
		 )

	if YOLO_TRT_QUANTIZE_MODE == 'INT8':
		converter.convert( calibration_input_fn = calibration_input_fn )
	else:
		converter.convert( )

	converter.summary( )

	#converter.build( )

	if not YOLO_CUSTOM_WEIGHTS:
		SAVED_TRT_MODEL = f"./{YOLO_CHECKPOINTS}/{YOLO_TYPE}{'-tiny' if YOLO_TINY else ''}-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}"
	else:
		SAVED_TRT_MODEL = f"{CUSTOM_WEIGHTS}-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}"
	converter.save( output_saved_model_dir = output_saved_model_dir )
	print( f'Done converting to TensorRT ( {quantize_mode} ), model saved to:', repr( output_saved_model_dir ) )

convert_to_trt( 'FP32' )
convert_to_trt( 'FP16' )
convert_to_trt( 'INT8' )
