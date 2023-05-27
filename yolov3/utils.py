###############################################################################
#
#	utils.py		Common YOLO functions
#
#
#  Portions Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

from multiprocessing import Process, Manager, Queue, current_process
#from multiprocessing import Process, Queue, current_process
#import queue
import sys
import time
import random
import colorsys
import numpy as np
import cv2
import tensorflow as tf
from yolov3.classes import *
from yolov3.yolov4 import *
from tensorflow.python.saved_model import tag_constants

QUEUE_SIZE = 20
PROCESS_TIMEOUT = 3
QUEUE_TIMEOUT = 10
EXIT_TIMEOUT = .5

TEXT_COLOR_BGR = ( 0, 0, 255 )

def load_yolo_weights( model, weights_file ):
	tf.keras.backend.clear_session( )	# used to reset layer names
	# load Darknet original weights to TensorFlow model
	if YOLO_TYPE == "yolov3":
		range1 = 75 if not YOLO_TINY else 13
		range2 = [ 58, 66, 74 ] if not YOLO_TINY else [ 9, 12 ]
	if YOLO_TYPE == "yolov4":
		range1 = 110 if not YOLO_TINY else 21
		range2 = [ 93, 101, 109 ] if not YOLO_TINY else [ 17, 20 ]

	with open( weights_file, 'rb' ) as wf:
		major, minor, revision, seen, _ = np.fromfile( wf, dtype = np.int32, count = 5 )

		j = 0
		for i in range( range1 ):
			if i > 0:
				conv_layer_name = 'conv2d_%d' %i
			else:
				conv_layer_name = 'conv2d'

			if j > 0:
				bn_layer_name = 'batch_normalization_%d' %j
			else:
				bn_layer_name = 'batch_normalization'

			conv_layer = model.get_layer( conv_layer_name )
			filters = conv_layer.filters
			k_size = conv_layer.kernel_size[ 0 ]
			in_dim = conv_layer.input_shape[ -1 ]

			if i not in range2:
				# darknet weights: [ beta, gamma, mean, variance ]
				bn_weights = np.fromfile( wf, dtype = np.float32, count = 4 * filters )
				# tf weights: [ gamma, beta, mean, variance ]
				bn_weights = bn_weights.reshape( ( 4, filters ) )[ [ 1, 0, 2, 3 ] ]
				bn_layer = model.get_layer( bn_layer_name )
				j += 1
			else:
				conv_bias = np.fromfile( wf, dtype = np.float32, count = filters )

			# darknet shape ( out_dim, in_dim, height, width )
			conv_shape = ( filters, in_dim, k_size, k_size )
			conv_weights = np.fromfile( wf, dtype = np.float32, count = np.product( conv_shape ) )
			# tf shape ( height, width, in_dim, out_dim )
			conv_weights = conv_weights.reshape( conv_shape ).transpose( [ 2, 3, 1, 0 ] )

			if i not in range2:
				conv_layer.set_weights( [ conv_weights ] )
				bn_layer.set_weights( bn_weights )
			else:
				conv_layer.set_weights( [ conv_weights, conv_bias ] )

		assert len( wf.read( ) ) == 0, 'failed to read all data'

def Load_Yolo_model( ):
	gpus = tf.config.experimental.list_physical_devices( 'GPU' )
	if len( gpus ) > 0:
		print( f'GPU: {gpus}' )
		try: tf.config.experimental.set_memory_growth( gpus[ 0 ], True )
		except RuntimeError: pass

	if YOLO_FRAMEWORK == "tf":	# using TensorFlow framework
		yolo = Create_Yolo( input_size = YOLO_INPUT_SIZE )
		if not YOLO_CUSTOM_WEIGHTS:
			load_yolo_weights( yolo, DARKNET_WEIGHTS )	# use Darknet weights
		else:
			yolo.load_weights( CUSTOM_WEIGHTS )		# use custom weights
	elif YOLO_FRAMEWORK == "trt":	# TensorRT detection
		saved_model_loaded = tf.saved_model.load( SAVED_TRT_MODEL, tags = [ tag_constants.SERVING ] )
		signature_keys = list( saved_model_loaded.signatures.keys( ) )
		yolo = saved_model_loaded.signatures[ 'serving_default' ]

	return yolo

def image_preprocess( image, target_size, gt_boxes = None ):
	ih, iw	= target_size
	h,  w, _  = image.shape

	scale = min( iw / w, ih / h )
	nw, nh  = int( scale * w ), int( scale * h )
	image_resized = cv2.resize( image, ( nw, nh ) )

	image_padded = np.full( shape = [ ih, iw, 3 ], fill_value = 128. )
	dw, dh = ( iw - nw ) // 2, ( ih - nh ) // 2
	image_padded[ dh : nh + dh, dw : nw + dw, : ] = image_resized
	image_padded /= 255.

	if gt_boxes is None:
		return image_padded
	else:
		gt_boxes[ :, [ 0, 2 ] ] = gt_boxes[ :, [ 0, 2 ] ] * scale + dw
		gt_boxes[ :, [ 1, 3 ] ] = gt_boxes[ :, [ 1, 3 ] ] * scale + dh
		return image_padded, gt_boxes


def draw_bbox( image, bboxes, show_label = True, show_confidence = True, Text_colors = ( 0, 0, 0 ), rectangle_colors = '', tracking = False ):
	image_h, image_w, _ = image.shape
	hsv_tuples = [ ( ( x + 1. ) / NUM_CLASSES, 1., 1. ) for x in range( NUM_CLASSES ) ]
#	print( "hsv_tuples", hsv_tuples )
	colors = list( map( lambda x: colorsys.hsv_to_rgb( *x ), hsv_tuples ) )
	colors = list( map( lambda x: ( int( x[ 0 ] * 255 ), int( x[ 1 ] * 255 ), int( x[ 2 ] * 255 ) ), colors ) )

#	random.seed( 0 )
#	random.shuffle( colors )
#	random.seed( None )

	for i, bbox in enumerate( bboxes ):
		coor = np.array( bbox[ : 4 ], dtype = np.int32 )
		score = bbox[ 4 ]
		class_ind = int( bbox[ 5 ] )
		bbox_color = rectangle_colors if rectangle_colors != '' else colors[ class_ind ]
		bbox_thick = int( ( 0.6 * ( image_h + image_w ) ) / 1000 )
		if bbox_thick < 1: bbox_thick = 1
		fontScale = 0.75 * bbox_thick
		( x1, y1 ), ( x2, y2 ) = ( int( coor[ 0 ] ), int( coor[ 1 ] ) ), ( int( coor[ 2 ] ), int( coor[ 3 ] ) )

		# put object rectangle
		cv2.rectangle( image, ( x1, y1 ), ( x2, y2 ), bbox_color, bbox_thick )

		if show_label:
			# get text label
			score_str = " {:.2f}".format( score ) if show_confidence else ""

			if tracking: score_str = " "+str( score )

			try:
				label = "{}".format( CLASS_NAMES[ class_ind ] ) + score_str
			except KeyError:
				print( "You received KeyError, this might be that you are trying to use yolo original weights" )
				print( "while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True" )

			# get text size
			( text_width, text_height ), baseline = cv2.getTextSize( label, cv2.FONT_HERSHEY_PLAIN,
				fontScale, thickness = bbox_thick // 2 )
			# put filled text rectangle
			cv2.rectangle( image, ( x1, y1 ), ( int( x1 + text_width ), int( y1 - text_height - baseline ) ), bbox_color, thickness = cv2.FILLED )

			# put text above rectangle
			cv2.putText( image, label, ( x1, y1 - 2 ), cv2.FONT_HERSHEY_PLAIN,
				fontScale, Text_colors, bbox_thick // 2, lineType = cv2.LINE_AA )

	return image


def bboxes_iou( boxes1, boxes2 ):
	boxes1		= np.array( boxes1 )
	boxes2		= np.array( boxes2 )

	boxes1_area	= ( boxes1[ ..., 2 ] - boxes1[ ..., 0 ] ) * ( boxes1[ ..., 3 ] - boxes1[ ..., 1 ] )
	boxes2_area	= ( boxes2[ ..., 2 ] - boxes2[ ..., 0 ] ) * ( boxes2[ ..., 3 ] - boxes2[ ..., 1 ] )

	left_up		= np.maximum( boxes1[ ..., : 2 ], boxes2[ ..., : 2 ] )
	right_down	= np.minimum( boxes1[ ..., 2 : ], boxes2[ ..., 2 : ] )

	inter_section	= np.maximum( right_down - left_up, 0.0 )
	inter_area	= inter_section[ ..., 0 ] * inter_section[ ..., 1 ]
	union_area	= boxes1_area + boxes2_area - inter_area
	ious		= np.maximum( 1.0 * inter_area / union_area, np.finfo( np.float32 ).eps )

	return ious


def nms( bboxes, iou_threshold, sigma = 0.3, method = 'nms' ):
	"""
	param bboxes: ( xmin, ymin, xmax, ymax, score, class )

	Note: soft-nms,	https://arxiv.org/pdf/1704.04503.pdf
			https://github.com/bharatsingh430/soft-nms
	"""
	classes_in_img = list( set( bboxes[ :, 5 ] ) )
	best_bboxes = [ ]

	for cls in classes_in_img:
		cls_mask = ( bboxes[ :, 5 ] == cls )
		cls_bboxes = bboxes[ cls_mask ]
		# Process 1: Determine whether the number of bounding boxes is greater than 0
		while len( cls_bboxes ) > 0:
			# Process 2: Select the bounding box with the highest score according to socre order A
			max_ind = np.argmax( cls_bboxes[ :, 4 ] )
			best_bbox = cls_bboxes[ max_ind ]
			best_bboxes.append( best_bbox )
			cls_bboxes = np.concatenate( [ cls_bboxes[ : max_ind ], cls_bboxes[ max_ind + 1 : ] ] )
			# Process 3: Calculate this bounding box A and
			# Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
			iou = bboxes_iou( best_bbox[ np.newaxis, : 4 ], cls_bboxes[ :, : 4 ] )
			weight = np.ones( ( len( iou ), ), dtype = np.float32 )

			assert method in [ 'nms', 'soft-nms' ]

			if method == 'nms':
				iou_mask = iou > iou_threshold
				weight[ iou_mask ] = 0.0

			if method == 'soft-nms':
				weight = np.exp( -( 1.0 * iou ** 2 / sigma ) )

			cls_bboxes[ :, 4 ] = cls_bboxes[ :, 4 ] * weight
			score_mask = cls_bboxes[ :, 4 ] > 0.
			cls_bboxes = cls_bboxes[ score_mask ]

	return best_bboxes


def postprocess_boxes( pred_bbox, original_image, input_size, score_threshold ):
	valid_scale = [ 0, np.inf ]
	pred_bbox = np.array( pred_bbox )

	pred_xywh = pred_bbox[ :, 0 : 4 ]
	pred_conf = pred_bbox[ :, 4 ]
	pred_prob = pred_bbox[ :, 5 : ]

	# 1. ( x, y, w, h ) --> ( xmin, ymin, xmax, ymax )
	pred_coor = np.concatenate( [ pred_xywh[ :, : 2 ] - pred_xywh[ :, 2 : ] * .5,
			pred_xywh[ :, : 2 ] + pred_xywh[ :, 2 : ] * .5 ], axis = -1 )
	# 2. ( xmin, ymin, xmax, ymax ) --> ( xmin_org, ymin_org, xmax_org, ymax_org )
	org_h, org_w = original_image.shape[ : 2 ]
	resize_ratio = min( input_size / org_w, input_size / org_h )

	dw = ( input_size - resize_ratio * org_w ) / 2
	dh = ( input_size - resize_ratio * org_h ) / 2

	pred_coor[ :, 0 : : 2 ] = 1.0 * ( pred_coor[ :, 0 : : 2 ] - dw ) / resize_ratio
	pred_coor[ :, 1 : : 2 ] = 1.0 * ( pred_coor[ :, 1 : : 2 ] - dh ) / resize_ratio

	# 3. clip some boxes those are out of range
	pred_coor = np.concatenate( [ np.maximum( pred_coor[ :, : 2 ], [ 0, 0 ] ),
			np.minimum( pred_coor[ :, 2 : ], [ org_w - 1, org_h - 1 ] ) ], axis = -1 )
	invalid_mask = np.logical_or( ( pred_coor[ :, 0 ] > pred_coor[ :, 2 ] ), ( pred_coor[ :, 1 ] > pred_coor[ :, 3 ] ) )
	pred_coor[ invalid_mask ] = 0

	# 4. discard some invalid boxes
	bboxes_scale = np.sqrt( np.multiply.reduce( pred_coor[ :, 2 : 4 ] - pred_coor[ :, 0 : 2 ], axis = -1 ) )
	scale_mask = np.logical_and( ( valid_scale[ 0 ] < bboxes_scale ), ( bboxes_scale < valid_scale[ 1 ] ) )

	# 5. discard boxes with low scores
	classes = np.argmax( pred_prob, axis = -1 )
	scores = pred_conf * pred_prob[ np.arange( len( pred_coor ) ), classes ]
	score_mask = scores > score_threshold
	mask = np.logical_and( scale_mask, score_mask )
	coors, scores, classes = pred_coor[ mask ], scores[ mask ], classes[ mask ]

	return np.concatenate( [ coors, scores[ :, np.newaxis ], classes[ :, np.newaxis ] ], axis = -1 )


def detect_image( Yolo, image_path, output_path, input_size = 416, show = False, score_threshold = 0.3, iou_threshold = 0.45, rectangle_colors = '' ):
	original_image = cv2.imread( image_path )
#	original_image = cv2.cvtColor( original_image, cv2.COLOR_BGR2RGB )
#	original_image = cv2.cvtColor( original_image, cv2.COLOR_BGR2RGB )

	image_data = image_preprocess( np.copy( original_image ), [ input_size, input_size ] )
	image_data = image_data[ np.newaxis, ... ].astype( np.float32 )

#	Yolo.training = False

	if YOLO_FRAMEWORK == "tf":
		pred_bbox = Yolo( image_data, training = False )
	#	pred_bbox = Yolo.predict( image_data, verbose = 0 )
	elif YOLO_FRAMEWORK == "trt":
		batched_input = tf.constant( image_data )
		result = Yolo( batched_input )
		pred_bbox = [ ]
		for key, value in result.items( ):
			value = value.numpy( )
			pred_bbox.append( value )

	pred_bbox = [ tf.reshape( x, ( -1, tf.shape( x )[ -1 ] ) ) for x in pred_bbox ]
	pred_bbox = tf.concat( pred_bbox, axis = 0 )

	bboxes = postprocess_boxes( pred_bbox, original_image, input_size, score_threshold )
	bboxes = nms( bboxes, iou_threshold, method = 'nms' )

#	original_image = cv2.cvtColor( original_image, cv2.COLOR_BGR2RGB )
	image = draw_bbox( original_image, bboxes, rectangle_colors = rectangle_colors )
#	CreateXMLfile( "XML_Detections", str( int( time.time( ) ) ), original_image, bboxes, CLASS_NAMES )

	if output_path: cv2.imwrite( output_path, image )
	if show:
		# Show the image
		cv2.imshow( "predicted image", image )
		# Load and hold the image
		if cv2.waitKey( 0 ) & 0xFF == ord( 'q' ):
			return None
		# To close the window after the required kill value was provided
	#	cv2.destroyAllWindows( )

	return image

def Predict_bbox_mp( Stop_in, Stop, Frames_data, Predicted_data, Processing_times, realtime ):
	print( f"Predict_bbox_mp: {current_process( ).name}", flush = True )

	gpus = tf.config.experimental.list_physical_devices( 'GPU' )
	if len( gpus ) > 0:
		try: tf.config.experimental.set_memory_growth( gpus[ 0 ], True )
		except RuntimeError: pass

	Yolo = Load_Yolo_model( )

#	Yolo.training = False

	tm = -1.
	frame = False

	while True:
		if not Stop_in.empty( ):
			if Stop.empty( ):
				print( "Predict_bbox_mp: Stop signaled...", flush = True )
				Stop.put( 1 )
			if tm >= 0. and time.perf_counter( ) - tm > PROCESS_TIMEOUT:
				break

		"""
		ATTENTION!
		Queue.empty( ) implementation in the queue module (and multiprocessing.Queue too) seems to be broken, and this code omits it,
		but Queue.qsize( ) may be not implemented on MacOS.
		Queue from multiprocessing.Manager( ) looks working and is in use now, although it's significantly slower.
		"""
	#	while Frames_data.qsize( ) if realtime else not frame:
	#		try:
	#			image_data = Frames_data.get_nowait( )
	#			frame = True
	#		except queue.Empty:
	#			frame = False
		if realtime:
			while not Frames_data.empty( ):
				image_data = Frames_data.get_nowait( )
				frame = True
		elif not frame and not Frames_data.empty( ):
			image_data = Frames_data.get_nowait( )
			frame = True

		if frame and not Processing_times.full( ) and not Predicted_data.full( ):
			tm = time.perf_counter( )

			Processing_times.put( tm )
			frame = False

			if YOLO_FRAMEWORK == "tf":
				pred_bbox = Yolo( image_data, training = False )
			#	pred_bbox = Yolo.predict( image_data, verbose = 0 )
			elif YOLO_FRAMEWORK == "trt":
				batched_input = tf.constant( image_data )
				result = Yolo( batched_input )
				pred_bbox = [ ]
				for key, value in result.items( ):
					value = value.numpy( )
					pred_bbox.append( value )

			pred_bbox = [ tf.reshape( x, ( -1, tf.shape( x )[ -1 ] ) ) for x in pred_bbox ]
			pred_bbox = tf.concat( pred_bbox, axis = 0 )

			Predicted_data.put( pred_bbox )

		#	print( "Predict_bbox_mp: {:.3f}".format( time.perf_counter( ) - tm ), flush = True )
		#	print( "\t", Processing_times.qsize( ), "\t\t", Frames_data.qsize( ), "\t", Predicted_data.qsize( ), flush = True )

	print( "Predict_bbox_mp: End...", flush = True )

def Postprocess_mp( Stop_in, Stop, Original_frames, Frames_data, Predicted_data, Processed_data, Processing_times, input_size, score_threshold, iou_threshold, fps, realtime, rectangle_colors ):
	print( f"Postprocess_mp: {current_process( ).name}", flush = True )

	times = [ ]
	tm = -1.
	processing = False

	while True:
		if not Stop_in.empty( ):
			if Stop.empty( ):
				print( "Postprocess_mp: Stop signaled...", flush = True )
				Stop.put( 1 )
			if tm >= 0. and time.perf_counter( ) - tm > PROCESS_TIMEOUT:
				break

		if not realtime and Predicted_data.empty( ):
			continue

		if not processing:
			if not Processing_times.empty( ):
				tp = Processing_times.get_nowait( )
				processing = True
			else:
				continue

		if not Predicted_data.empty( ):
			pred_bbox = Predicted_data.get_nowait( )

			tm = time.perf_counter( )
			processing = False

		if ( not processing or not Original_frames.empty( ) ) and not Processed_data.full( ):
			if not Original_frames.empty( ):
				original_image = Original_frames.get_nowait( )

			if tm >= 0.:
				image = original_image.copy( )
				bboxes = postprocess_boxes( pred_bbox, image, input_size, score_threshold )
				bboxes = nms( bboxes, iou_threshold, method = 'nms' )
				draw_bbox( image, bboxes, rectangle_colors = rectangle_colors )

				if not processing:
					times.append( tm - tp )
					times = times[ -30 : ]
					ms = sum( times ) / len( times ) * 1000
					perf = 1000 / ms
				#	print( "Time: {:.2f} ms, Final FPS: {:.1f}".format( ms, perf ), flush = True )

				image = cv2.putText( image, "{:.1f} FPS".format( perf ), ( 6, image.shape[ 0 ] - 6 ), cv2.FONT_HERSHEY_PLAIN, 1, TEXT_COLOR_BGR, 2 )

				Processed_data.put( image )

			#	print( "Postprocess_mp: {:.3f}".format( time.perf_counter( ) - tm ), flush = True )
			#	print( "\t", Processing_times.qsize( ), "\t", Original_frames.qsize( ), "\t\t", Predicted_data.qsize( ), "\t", Processed_data.qsize( ), flush = True )

	print( "Postprocess_mp: End...", flush = True )

def Show_image_mp( Stop_in, Stop, Processed_data, output_path, width, height, fps, show ):
	print( f"Show_image_mp: {current_process( ).name}", flush = True )

	if output_path:
		codec = cv2.VideoWriter_fourcc( 'm', 'p', '4', 'v' )
		out = cv2.VideoWriter( output_path, codec, fps, ( width, height ) )

	tm = -1.

	while True:
		if not Processed_data.empty( ):
			tm = time.perf_counter( )

			while not Processed_data.empty( ):
				image = Processed_data.get_nowait( )
				if output_path: out.write( image )

			if show:
				cv2.imshow( 'output', image )
				if cv2.waitKey( 1 ) & 0xFF == ord( "q" ) and Stop.empty( ):
					print( "Show_image_mp: Terminating...", flush = True )
				#	cv2.destroyAllWindows( )
					Stop.put( 1 )

		#	print( "Show_image_mp: {:.3f}".format( time.perf_counter( ) - tm ), flush = True )
		#	print( "\t\t\t\t\t", Processed_data.qsize( ), flush = True )

		if not Stop_in.empty( ):
			if Stop.empty( ):
				print( "Show_image_mp: Stop signaled...", flush = True )
				Stop.put( 1 )
			if tm >= 0. and time.perf_counter( ) - tm > PROCESS_TIMEOUT:
				break

	print( "Show_image_mp: End...", flush = True )
	if output_path: out.release( )
	if show: cv2.destroyAllWindows( )

def Get_image_mp( Stop_in, Stop, Original_frames, Frames_data, generator, video_path, input_size, width, height, fps, realtime ):
	print( f"Get_image_mp: {current_process( ).name}", flush = True )

	if not generator:
		if realtime:#not video_path:
			if sys.platform == "win32":
				vid = cv2.VideoCapture( 0, cv2.CAP_DSHOW )
			else:
				vid = cv2.VideoCapture( 0 )
		#	vid.set( cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc( 'm', 'j', 'p', 'g' ) )
			vid.set( cv2.CAP_PROP_FPS, fps )
			vid.set( cv2.CAP_PROP_FRAME_WIDTH, width )
			vid.set( cv2.CAP_PROP_FRAME_HEIGHT, height )
		else:
			vid = cv2.VideoCapture( video_path )
	else:
		vid = generator( width, height, fps )

	tm = -1.

	while True:
		free = not Original_frames.full( ) and not Frames_data.full( )

		if free or realtime:
			ret, original_image = vid.read( )
			if not ret:
				print( "Get_image_mp: End of input video stream...", flush = True )
				if Stop.empty( ):
					Stop.put( 1 )
				if tm < 0.: sys.exit( 1 )
				break

			tm = time.perf_counter( )

			if free:
		#		print( "Get_image_mp: Processing frame @{:.2f}".format( tm ), flush = True )
				Original_frames.put( original_image )
				image_data = image_preprocess( np.copy( original_image ), [ input_size, input_size ] )
				image_data = image_data[ np.newaxis, ... ].astype( np.float32 )
				Frames_data.put( image_data )
		#	else:
		#		print( "Get_image_mp: Skipping frame @{:.2f}".format( tm ), flush = True )

		if not Stop_in.empty( ):
			print( "Get_image_mp: Stop signaled...", flush = True )
			if Stop.empty( ):
				Stop.put( 1 )
			break

	vid.release( )

	print( "Get_image_mp: End...", flush = True )

# detect from real-time video source
def detect_video_realtime_mp( generator, video_path, output_path, input_size = 416, width = 640, height = 480, fps = 30, show = False, score_threshold = 0.3, iou_threshold = 0.45, realtime = False, rectangle_colors = '' ):
	if not generator:
		if realtime:#not video_path:
			if sys.platform == "win32":
				vid = cv2.VideoCapture( 0, cv2.CAP_DSHOW )
			else:
				vid = cv2.VideoCapture( 0 )
		#	vid.set( cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc( 'm', 'j', 'p', 'g' ) )
			vid.set( cv2.CAP_PROP_FPS, fps )
			vid.set( cv2.CAP_PROP_FRAME_WIDTH, width )
			vid.set( cv2.CAP_PROP_FRAME_HEIGHT, height )
		else:
			vid = cv2.VideoCapture( video_path )
		# by default VideoCapture returns float instead of int
		width = int( vid.get( cv2.CAP_PROP_FRAME_WIDTH ) )
		height = int( vid.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
		fps = int( vid.get( cv2.CAP_PROP_FPS ) )
		vid.release( )

	print( f"***INFO: Video format: {width} x {height}, {fps} FPS" )

	"""
	ATTENTION!
	Queue.empty( ) implementation in the queue module and multiprocessing.Queue seems to be broken and Queue.qsize( ) may be not on MacOS.
	Queue from multiprocessing.Manager( ) looks working and is in use now, although it's significantly slower.
	"""
#	Stop_in = Queue( 1 )
#	Stop = Queue( 1 )
#	Stop_postprocess = Queue( 1 )
#	Stop_show = Queue( 1 )
#	Original_frames = Queue( QUEUE_SIZE )
#	Frames_data = Queue( QUEUE_SIZE )
#	Predicted_data = Queue( QUEUE_SIZE )
#	Processed_data = Queue( QUEUE_SIZE )
#	Processing_times = Queue( QUEUE_SIZE )
	m = Manager( )
	Stop_in = m.Queue( 1 )
	Stop = m.Queue( 1 )
	Stop_postprocess = m.Queue( 1 )
	Stop_show = m.Queue( 1 )
	Original_frames = m.Queue( QUEUE_SIZE )
	Frames_data = m.Queue( QUEUE_SIZE )
	Predicted_data = m.Queue( QUEUE_SIZE )
	Processed_data = m.Queue( QUEUE_SIZE )
	Processing_times = m.Queue( QUEUE_SIZE )

	p0 = Process( target = Get_image_mp, name = "Get_image_mp",
		args = ( Stop_in, Stop, Original_frames, Frames_data, generator, video_path, input_size, width, height, fps, realtime ) )
	p1 = Process( target = Predict_bbox_mp, name = "YOLO",
		args = ( Stop, Stop_postprocess, Frames_data, Predicted_data, Processing_times, realtime ) )
	p2 = Process( target = Postprocess_mp, name = "Postprocess_mp",
		args = ( Stop_postprocess, Stop_show, Original_frames, Frames_data, Predicted_data, Processed_data, Processing_times, input_size, score_threshold, iou_threshold, fps, realtime, rectangle_colors ) )
	p3 = Process( target = Show_image_mp, name = "Show_image_mp",
		args = ( Stop_show, Stop_in, Processed_data, output_path, width, height, fps, show ) )
	p0.start( )
	p1.start( )
	p2.start( )
	p3.start( )

	while True:
		time.sleep( 1 )

		if not p0.is_alive( ) or not p1.is_alive( ) or not p2.is_alive( ) or not p3.is_alive( ):
			break

	tm = time.perf_counter( )

	while True:
		time.sleep( 1 )

		if p0.exitcode or not p1.is_alive( ) and not p2.is_alive( ) and not p3.is_alive( ):
			break
		elif not Processed_data.empty( ):
			tm = time.perf_counter( )
		elif time.perf_counter( ) - tm > QUEUE_TIMEOUT:
			break

	if p0.is_alive( ) or p1.is_alive( ) or p2.is_alive( ) or p3.is_alive( ):
		time.sleep( 1 )

	print( "Remains:", p0.is_alive( ), p1.is_alive( ), p2.is_alive( ), p3.is_alive( ), flush = True )
	print( "\t", Original_frames.qsize( ), Frames_data.qsize( ), Predicted_data.qsize( ), Processed_data.qsize( ), flush = True )

	if p0.is_alive( ): p0.terminate( ); time.sleep( EXIT_TIMEOUT )
	if p1.is_alive( ): p1.terminate( ); time.sleep( EXIT_TIMEOUT )
	if p2.is_alive( ): p2.terminate( ); time.sleep( EXIT_TIMEOUT )
	if p3.is_alive( ):
		p3.terminate( ); time.sleep( EXIT_TIMEOUT );
		if show: cv2.destroyAllWindows( )

#	print( "Remains:", p0.is_alive( ), p1.is_alive( ), p2.is_alive( ), p3.is_alive( ), flush = True )

# detect from video file
def detect_video( Yolo, video_path, output_path, input_size = 416, show = False, score_threshold = 0.3, iou_threshold = 0.45, rectangle_colors = '' ):
	times, times_2 = [ ], [ ]

	vid = cv2.VideoCapture( video_path )
	# by default VideoCapture returns float instead of int
	width = int( vid.get( cv2.CAP_PROP_FRAME_WIDTH ) )
	height = int( vid.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
	fps = int( vid.get( cv2.CAP_PROP_FPS ) )

	print( f"***INFO: Video format: {width} x {height}, {fps} FPS" )

	if output_path:
		codec = cv2.VideoWriter_fourcc( 'm', 'p', '4', 'v' )
		out = cv2.VideoWriter( output_path, codec, fps, ( width, height ) )

#	Yolo.training = False

	while True:
		ret, original_image = vid.read( )
		if not ret:
			break

		image_data = image_preprocess( np.copy( original_image ), [ input_size, input_size ] )
		image_data = image_data[ np.newaxis, ... ].astype( np.float32 )

		t1 = time.perf_counter( )

		if YOLO_FRAMEWORK == "tf":
			pred_bbox = Yolo( image_data, training = False )
		#	pred_bbox = Yolo.predict( image_data, verbose = 0 )
		elif YOLO_FRAMEWORK == "trt":
			batched_input = tf.constant( image_data )
			result = Yolo( batched_input )
			pred_bbox = [ ]
			for key, value in result.items( ):
				value = value.numpy( )
				pred_bbox.append( value )

		t2 = time.perf_counter( )

		pred_bbox = [ tf.reshape( x, ( -1, tf.shape( x )[ -1 ] ) ) for x in pred_bbox ]
		pred_bbox = tf.concat( pred_bbox, axis = 0 )

		bboxes = postprocess_boxes( pred_bbox, original_image, input_size, score_threshold )
		bboxes = nms( bboxes, iou_threshold, method = 'nms' )
		image = draw_bbox( original_image, bboxes, rectangle_colors = rectangle_colors )

		t3 = time.perf_counter( )
		times.append( t2 - t1 )
		times_2.append( t3 - t1 )

		times = times[ -30 : ]
		times_2 = times_2[ -30 : ]

		ms = sum( times ) / len( times ) * 1000
		perf = 1000 / ms
		perf2 = 1000 / ( sum( times_2 ) / len( times_2 ) * 1000 )

		image = cv2.putText( image, "{:.1f} FPS".format( perf ), ( 6, image.shape[ 0 ] - 6 ), cv2.FONT_HERSHEY_PLAIN, 1, TEXT_COLOR_BGR, 2 )
	#	CreateXMLfile( "XML_Detections", str( int( time.time( ) ) ), original_image, bboxes, CLASS_NAMES )
	#	print( "Time: {:.2f} ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format( ms, perf, perf2 ) )

		if output_path: out.write( image )
		if show:
			cv2.imshow( 'output', image )
			if cv2.waitKey( 1 ) & 0xFF == ord( "q" ):
				break

	vid.release( )
	if output_path: out.release( )
	if show: cv2.destroyAllWindows( )

# detect from real-time video source
def detect_realtime( Yolo, generator, output_path, input_size = 416, width = 640, height = 480, fps = 30, show = False, score_threshold = 0.3, iou_threshold = 0.45, rectangle_colors = '' ):
	times = [ ]

	if not generator:
		if sys.platform == "win32":
			vid = cv2.VideoCapture( 0, cv2.CAP_DSHOW )
		else:
			vid = cv2.VideoCapture( 0 )
	#	vid.set( cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc( 'm', 'j', 'p', 'g' ) )
		vid.set( cv2.CAP_PROP_FPS, fps )
		vid.set( cv2.CAP_PROP_FRAME_WIDTH, width )
		vid.set( cv2.CAP_PROP_FRAME_HEIGHT, height )
		# by default VideoCapture returns float instead of int
		width = int( vid.get( cv2.CAP_PROP_FRAME_WIDTH ) )
		height = int( vid.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
		fps = int( vid.get( cv2.CAP_PROP_FPS ) )
	else:
		vid = generator( width, height, fps )

	print( f"***INFO: Video format: {width} x {height}, {fps} FPS" )

	if output_path:
		codec = cv2.VideoWriter_fourcc( 'm', 'p', '4', 'v' )
		out = cv2.VideoWriter( output_path, codec, fps, ( width, height ) )

#	Yolo.training = False

	while True:
		ret, original_frame = vid.read( )
		if not ret:
			break

		image_data = image_preprocess( np.copy( original_frame ), [ input_size, input_size ] )
		image_data = image_data[ np.newaxis, ... ].astype( np.float32 )

		t1 = time.perf_counter( )

		if YOLO_FRAMEWORK == "tf":
			pred_bbox = Yolo( image_data, training = False )
		#	pred_bbox = Yolo.predict( image_data, verbose = 0 )
		elif YOLO_FRAMEWORK == "trt":
			batched_input = tf.constant( image_data )
			result = Yolo( batched_input )
			pred_bbox = [ ]
			for key, value in result.items( ):
				value = value.numpy( )
				pred_bbox.append( value )

		t2 = time.perf_counter( )

		pred_bbox = [ tf.reshape( x, ( -1, tf.shape( x )[ -1 ] ) ) for x in pred_bbox ]
		pred_bbox = tf.concat( pred_bbox, axis = 0 )

		bboxes = postprocess_boxes( pred_bbox, original_frame, input_size, score_threshold )
		bboxes = nms( bboxes, iou_threshold, method = 'nms' )
		frame = draw_bbox( original_frame, bboxes, rectangle_colors = rectangle_colors )

		times.append( t2 - t1 )
		times = times[ -30 : ]

		ms = sum( times ) / len( times ) * 1000
		perf = 1000 / ms

		image = cv2.putText( frame, "{:.1f} FPS".format( perf ), ( 6, frame.shape[ 0 ] - 6 ), cv2.FONT_HERSHEY_PLAIN, 1, TEXT_COLOR_BGR, 2 )
	#	CreateXMLfile( "XML_Detections", str( int( time.time( ) ) ), original_frame, bboxes, CLASS_NAMES )
	#	print( "Time: {:.2f} ms, {:.1f} FPS".format( ms, perf ) )

		if output_path: out.write( frame )
		if show:
			cv2.imshow( 'output', frame )
			if cv2.waitKey( 1 ) & 0xFF == ord( "q" ):
				break

	vid.release( )
	if output_path: out.release( )
	if show: cv2.destroyAllWindows( )
