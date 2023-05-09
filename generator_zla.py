###############################################################################
#
#	generator_zla.py	Video generator for real-time detection testing
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

#import set_working_directory
import os
#os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
import cv2
import numpy as np
import random
import time
#from yolov3.classes import *
import input_data

class generator( object ):
	OBJECTS = 6
	SIZE = 28
	LIVE = 10
	SPEED = 10
	RATIOS = [ 0.4, 4. ]
	SIZES = [ 0.1, RATIOS[ 1 ] / 6 ]

	def __init__( self, width, height, fps = 25, no_of_frames = 0 ):
		self.width = width
		self.height = height
		self.fps = fps
		self.no_of_frames = no_of_frames
		self.time = time.perf_counter( )
		self.datasets = input_data.read_data_sets( train_dir = "mnist" )
		self.dataset = self.datasets.test
		self.datasize = self.datasets.test.num_examples
		self.ds_images, self.ds_labels = self.dataset.next_batch( self.datasize )
		self.objects, _ = self.dataset.next_batch( self.OBJECTS )
		self.lives = np.random.randint( 1, self.LIVE, self.OBJECTS, dtype = int )
		self.ratio = np.full( self.OBJECTS, 0.1, dtype = float )
		self.moves = np.random.randint( 1, self.SPEED, self.OBJECTS * 2, dtype = int )
		mask = np.random.random( self.OBJECTS * 2 ) < 0.5
		np.negative( self.moves, where = mask, out = self.moves )
		self.moves.shape = ( self.OBJECTS, 2 )
		self.sizes = np.random.uniform( self.SIZES[ 0 ], self.SIZES[ 1 ], self.OBJECTS )
	#	mask = np.random.random( self.OBJECTS ) < 0.5
	#	np.negative( self.sizes, where = mask, out = self.sizes )
		self.pos = np.array( [ [ random.random( ) * height, random.random( ) * width ] for _ in range( self.OBJECTS ) ], dtype = int )

	def release( self ):
		pass

	def __len__( self ):
		return self.no_of_frames

	def __iter__( self ):
		return self

	def __next__( self ):
		return read( )

	def new( self, i ):
		self.objects[ i ], _ = self.dataset.next_batch( 1 )
		self.lives[ i ] = np.random.randint( 1, self.LIVE, dtype = int )
		self.ratio[ i ] = 0.1
		moves = np.random.randint( 1, self.SPEED, 2, dtype = int )
		mask = np.random.random( 2 ) < 0.5
		np.negative( moves, where = mask, out = moves )
		self.moves[ i ] = moves
		self.sizes[ i ] = np.random.uniform( self.SIZES[ 0 ], self.SIZES[ 1 ] )
		self.pos[ i ] = np.array( [ random.random( ) * self.height, random.random( ) * self.width ], dtype = int )

	def read( self ):
		blank = np.ones( shape = [ self.height, self.width, 3 ], dtype = np.uint8 ) * 255
		for i in range( self.OBJECTS ):
			# Reshape into 2D array
			cv_image = self.objects[ i ].reshape( ( self.SIZE, self.SIZE ) ).astype( np.uint8 )
			cv_image = cv2.resize( cv_image, ( int( self.SIZE * self.ratio[ i ] ), int( self.SIZE * self.ratio[ i ] ) ) )
			h, w = cv_image.shape
			locs = np.array( np.where( cv_image != 255 ), dtype = int )
			mask = ( locs[ 0 ] - h // 2 + self.pos[ i ][ 0 ] >= 0 ) & ( locs[ 1 ] - w // 2 + self.pos[ i ][ 1 ] >= 0 ) & (
				locs[ 0 ] - h // 2 + self.pos[ i ][ 0 ] < self.height ) & ( locs[ 1 ] - w // 2 + self.pos[ i ][ 1 ] < self.width )
			locs_y = locs[ 0 ][ mask ]
			locs_x = locs[ 1 ][ mask ]
			blank[ locs_y - h // 2 + self.pos[ i ][ 0 ], locs_x - w // 2 + self.pos[ i ][ 1 ] ] = cv_image[ locs_y, locs_x, None ]

		self.ratio += self.sizes
		for i in np.nonzero( ( self.ratio > self.RATIOS[ 1 ] ) | ( self.ratio < self.RATIOS[ 0 ] ) )[ 0 ]:
			self.sizes[ i ] = -self.sizes[ i ]
			if self.ratio[ i ] < self.RATIOS[ 0 ]:
				self.ratio[ i ] = 2 * self.RATIOS[ 0 ] - self.ratio[ i ]
			else:
				self.ratio[ i ] = 2 * self.RATIOS[ 1 ] - self.ratio[ i ]

		self.pos += self.moves

	#	for i in range( self.OBJECTS ):
	#		bounce = False
	#		if self.pos[ i, 0 ] < 0 or self.pos[ i, 0 ] >= self.height:
	#			self.moves[ i, 0 ] = -self.moves[ i, 0 ]
	#			bounce = True
	#		if self.pos[ i, 1 ] < 0 or self.pos[ i, 1 ] >= self.width:
	#			self.moves[ i, 1 ] = -self.moves[ i, 1 ]
	#			bounce = True
	#		if bounce:
	#			if self.lives[ i ]:
	#				self.lives[ i ] -= 1
	#			else:
	#				self.new( i )
		mask = np.array( [ ( self.pos[ i, 0 ] < 0 or self.pos[ i, 0 ] >= self.height, self.pos[ i, 1 ] < 0 or self.pos[ i, 1 ] >= self.width ) for i in range( self.OBJECTS ) ], dtype = bool )
		np.negative( self.moves, where = mask, out = self.moves )
		mask = np.nonzero( mask )[ 0 ]
		for i in range( len( np.unique( mask ) ) ):
			if self.lives[ mask[ i ] ]:
				self.lives[ mask[ i ] ] -= 1
			else:
				self.new( mask[ i ] )

		while True:
			tm = time.perf_counter( )
			if tm - self.time >= 1. / self.fps:
				self.time = tm
				break
		return ( True, blank )
