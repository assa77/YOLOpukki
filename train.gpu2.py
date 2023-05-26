###############################################################################
#
#	train.gpu1.py		Trains YOLO model on custom datasets (using
#				GPU #2)
#
#  Portions Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

GPU = 2

#import set_working_directory
import os
import time
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '{}'.format(GPU)
os.environ[ 'TF_FORCE_GPU_ALLOW_GROWTH' ] = 'true'
from tensorflow.python.client import device_lib
print( device_lib.list_local_devices( ) )
import shutil
import numpy as np
import tensorflow as tf
#from tensorflow.keras.utils import plot_model
from yolov3.classes import *
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo, compute_loss
from yolov3.utils import load_yolo_weights
from evaluate_mAP import get_mAP

if YOLO_TINY: TRAIN_MODEL_NAME += "_Tiny"

def main( ):
	global TRAIN_FROM_CHECKPOINT

	gpus = tf.config.experimental.list_physical_devices( 'GPU' )
	print( f'GPU: {gpus}' )
	if len( gpus ) > 0:
		try: tf.config.experimental.set_memory_growth( gpus[ 0 ], True )
		except RuntimeError: pass

	if os.path.exists( TRAIN_LOGDIR ):
		shutil.rmtree( TRAIN_LOGDIR )
		time.sleep( 2 )
	writer = tf.summary.create_file_writer( TRAIN_LOGDIR )

	trainset = Dataset( 'train' )
	testset = Dataset( 'test' )

	steps_per_epoch = len( trainset )
	global_steps = tf.Variable( 0, trainable = False, dtype = tf.int64 )
	warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
	total_steps = TRAIN_EPOCHS * steps_per_epoch

	if TRAIN_TRANSFER:
		Darknet = Create_Yolo( input_size = YOLO_INPUT_SIZE, num_classes = len( read_class_names( YOLO_CLASSES ) ) )
		load_yolo_weights( Darknet, DARKNET_WEIGHTS )	# use Darknet weights

	yolo = Create_Yolo( input_size = YOLO_INPUT_SIZE, training = True )
	if TRAIN_FROM_CHECKPOINT:
		try:
			yolo.load_weights( f"./{YOLO_CHECKPOINTS}/{TRAIN_MODEL_NAME}" )
		except ValueError:
			print( "Shapes are incompatible, transfering Darknet weights" )
			TRAIN_FROM_CHECKPOINT = False

	if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
		for i, l in enumerate( Darknet.layers ):
			layer_weights = l.get_weights( )
			if layer_weights != [ ]:
				try:
					yolo.layers[ i ].set_weights( layer_weights )
				except:
					print( "skipping", yolo.layers[ i ].name )

	optimizer = tf.keras.optimizers.Adam( )

	def train_step( image_data, target ):
		with tf.GradientTape( ) as tape:
			pred_result = yolo( image_data, training = True )
			giou_loss = conf_loss = prob_loss = 0

			# optimizing process
			grid = 3 if not YOLO_TINY else 2
			for i in range( grid ):
				conv, pred = pred_result[ i * 2 ], pred_result[ i * 2 + 1 ]
				loss_items = compute_loss( pred, conv, *target[ i ], i )
				giou_loss += loss_items[ 0 ]
				conf_loss += loss_items[ 1 ]
				prob_loss += loss_items[ 2 ]

			total_loss = giou_loss + conf_loss + prob_loss

			gradients = tape.gradient( total_loss, yolo.trainable_variables )
			optimizer.apply_gradients( zip( gradients, yolo.trainable_variables ) )

			# update learning rate
			# about warmup: https://arxiv.org/pdf/1812.01187.pdf
			global_steps.assign_add( 1 )
			if global_steps <= warmup_steps:# and not TRAIN_TRANSFER:
				lr = global_steps / warmup_steps * TRAIN_LR_INIT
			else:
				lr = TRAIN_LR_END + 0.5 * ( TRAIN_LR_INIT - TRAIN_LR_END ) * (
					( 1 + tf.cos( ( global_steps - warmup_steps ) / ( total_steps - warmup_steps ) * np.pi ) ) )
			optimizer.lr.assign( lr.numpy( ) )

			# writing summary data
			with writer.as_default( ):
				tf.summary.scalar( "lr", optimizer.lr, step = global_steps )
				tf.summary.scalar( "loss/total_loss", total_loss, step = global_steps )
				tf.summary.scalar( "loss/giou_loss", giou_loss, step = global_steps )
				tf.summary.scalar( "loss/conf_loss", conf_loss, step = global_steps )
				tf.summary.scalar( "loss/prob_loss", prob_loss, step = global_steps )
			writer.flush( )

		return global_steps.numpy( ), optimizer.lr.numpy( ), giou_loss.numpy( ), conf_loss.numpy( ), prob_loss.numpy( ), total_loss.numpy( )

	validate_writer = tf.summary.create_file_writer( TRAIN_LOGDIR )
	def validate_step( image_data, target ):
		with tf.GradientTape( ) as tape:
			pred_result = yolo( image_data, training = False )
			giou_loss = conf_loss = prob_loss = 0

			# optimizing process
			grid = 3 if not YOLO_TINY else 2
			for i in range( grid ):
				conv, pred = pred_result[ i * 2 ], pred_result[ i * 2 + 1 ]
				loss_items = compute_loss( pred, conv, *target[ i ], i )
				giou_loss += loss_items[ 0 ]
				conf_loss += loss_items[ 1 ]
				prob_loss += loss_items[ 2 ]

			total_loss = giou_loss + conf_loss + prob_loss

		return giou_loss.numpy( ), conf_loss.numpy( ), prob_loss.numpy( ), total_loss.numpy( )

	mAP_model = Create_Yolo( input_size = YOLO_INPUT_SIZE )	# create second model to measure mAP

	best_val_loss = 1000	# should be large at start
	for epoch in range( TRAIN_EPOCHS ):
		start = time.time( )
		cur_time = time.strftime( "%H:%M:%S", time.localtime( start ) )
		cur_step = 0
		pperc = -1

		for image_data, target in trainset:
			results = train_step( image_data, target )
			cur_step += 1
			perc = ( cur_step * 1000 ) // ( steps_per_epoch + len( testset ) )
			if perc != pperc:
				print( "%s: Epoch %3d: %d.%d%%\r" %( cur_time, epoch, perc // 10, perc % 10 ), end = '' )
				pperc = perc

		if len( testset ) == 0:
			print( "configure TEST options to validate model" )
			yolo.save_weights( os.path.join( YOLO_CHECKPOINTS, TRAIN_MODEL_NAME ) )
			continue

		count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
		for image_data, target in testset:
			results = validate_step( image_data, target )
			count += 1
			giou_val += results[ 0 ]
			conf_val += results[ 1 ]
			prob_val += results[ 2 ]
			total_val += results[ 3 ]
			cur_step += 1
			perc = ( cur_step * 1000 ) // ( steps_per_epoch + len( testset ) )
			if perc != pperc:
				print( "%s: Epoch %3d: %d.%d%%\r" %( cur_time, epoch, perc // 10, perc % 10 ), end = '' )
				pperc = perc

		epoch_time = int( time.time( ) - start )

		# writing validate summary data
		with validate_writer.as_default( ):
			tf.summary.scalar( "validate_loss/total_val", total_val / count, step = epoch )
			tf.summary.scalar( "validate_loss/giou_val", giou_val / count, step = epoch )
			tf.summary.scalar( "validate_loss/conf_val", conf_val / count, step = epoch )
			tf.summary.scalar( "validate_loss/prob_val", prob_val / count, step = epoch )
		validate_writer.flush( )

		print( "{:02d}:{:02d}:{:02d}: Epoch {:3d}: giou_val_loss={:7.2f}, conf_val_loss={:7.2f}, prob_val_loss={:7.2f}, total_val_loss={:7.2f}"
			.format( epoch_time // 3600, epoch_time % 3600 // 60, epoch_time % 60,
			epoch, giou_val / count, conf_val / count, prob_val / count, total_val / count ) )

		if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
			save_directory = os.path.join( YOLO_CHECKPOINTS, TRAIN_MODEL_NAME + "_val_loss_{:7.2f}".format( total_val / count ) )
			yolo.save_weights( save_directory )
		if TRAIN_SAVE_BEST_ONLY and best_val_loss > total_val / count:
			save_directory = os.path.join( YOLO_CHECKPOINTS, TRAIN_MODEL_NAME )
			yolo.save_weights( save_directory )
			best_val_loss = total_val / count
		if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
			save_directory = os.path.join( YOLO_CHECKPOINTS, TRAIN_MODEL_NAME )
			yolo.save_weights( save_directory )

	# measure mAP of trained custom model
	try:
		mAP_model.load_weights( save_directory )	# use keras weights
		get_mAP( mAP_model, testset, score_threshold = TEST_SCORE_THRESHOLD, iou_threshold = TEST_IOU_THRESHOLD )
	except UnboundLocalError:
		print( "You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY and TRAIN_SAVE_CHECKPOINT lines in configs.py" )

if __name__ == '__main__':
	try:
		with tf.device( '/device:GPU:{}'.format( GPU ) ): main( )
	except RuntimeError as e:
		print( "***ERROR:", e )
