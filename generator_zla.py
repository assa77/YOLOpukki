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
import random
import time
import math
import numpy as np
import cv2
#from yolov3.classes import *
import input_data

def rotate( image, angle, scale = 1.0, border = ( 255, 255, 255 ) ):
	h, w = image.shape[ : 2 ]
	center = ( w / 2, h / 2 )
	matrix = cv2.getRotationMatrix2D( center, angle, scale )
	rad = math.radians( angle )
	sin = math.sin( rad )
	cos = math.cos( rad )
	b_w = int( ( h * abs( sin ) + w * abs( cos ) ) * scale )
	b_h = int( ( h * abs( cos ) + w * abs( sin ) ) * scale )

	matrix[ 0, 2 ] += ( ( b_w / 2 ) - center[ 0 ] )
	matrix[ 1, 2 ] += ( ( b_h / 2 ) - center[ 1 ] )

	return cv2.warpAffine( image, matrix, ( b_w, b_h ), flags = cv2.INTER_LINEAR,
		borderMode = cv2.BORDER_CONSTANT, borderValue = border )

class generator( object ):
	OBJECTS = 6
	SIZE = 28
	LIVE = 10
	SPEED = 10
	ROT_SPEED_MIN = 1
	ROT_SPEED_MAX = 30
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
		self.lives = np.random.randint( 1, self.LIVE + 1, self.OBJECTS, dtype = int )
		self.ratio = np.full( self.OBJECTS, 0.1, dtype = float )
		self.moves = np.random.randint( 1, self.SPEED + 1, self.OBJECTS * 2, dtype = int )
		mask = np.random.random( self.OBJECTS * 2 ) < 0.5
		np.negative( self.moves, where = mask, out = self.moves )
		self.moves.shape = ( self.OBJECTS, 2 )
		self.sizes = np.random.uniform( self.SIZES[ 0 ], self.SIZES[ 1 ], self.OBJECTS )
	#	mask = np.random.random( self.OBJECTS ) < 0.5
	#	np.negative( self.sizes, where = mask, out = self.sizes )
		self.pos = np.array( [ [ random.random( ) * height, random.random( ) * width ] for _ in range( self.OBJECTS ) ], dtype = int )
		self.rot = np.zeros( self.OBJECTS, dtype = int )
		self.rots = np.random.randint( self.ROT_SPEED_MIN, self.ROT_SPEED_MAX + 1, self.OBJECTS, dtype = int )
		mask = np.random.random( self.OBJECTS ) < 0.5
		np.negative( self.rots, where = mask, out = self.rots )
		mask = np.random.random( self.OBJECTS ) < 0.3
		self.rots[ mask ] = 0

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
		self.lives[ i ] = np.random.randint( 1, self.LIVE + 1, dtype = int )
		self.ratio[ i ] = 0.1
		moves = np.random.randint( 1, self.SPEED + 1, 2, dtype = int )
		mask = np.random.random( 2 ) < 0.5
		np.negative( moves, where = mask, out = moves )
		self.moves[ i ] = moves
		self.sizes[ i ] = np.random.uniform( self.SIZES[ 0 ], self.SIZES[ 1 ] )
		self.pos[ i ] = np.array( [ random.random( ) * self.height, random.random( ) * self.width ], dtype = int )
		self.rot[ i ] = 0
		self.rots[ i ] = 0 if random.random( ) < 0.3 else np.random.randint( self.ROT_SPEED_MIN, self.ROT_SPEED_MAX + 1, dtype = int )
		if random.random( ) < 0.5: self.rots[ i ] = -self.rots[ i ]

	def read( self ):
		blank = np.ones( shape = [ self.height, self.width, 3 ], dtype = np.uint8 ) * 255
		for i in range( self.OBJECTS ):
			# Reshape into 2D array
			cv_image = self.objects[ i ].reshape( ( self.SIZE, self.SIZE ) ).astype( np.uint8 )
		#	cv_image = cv2.resize( cv_image, ( int( self.SIZE * self.ratio[ i ] ), int( self.SIZE * self.ratio[ i ] ) ) )
			cv_image = rotate( cv_image, self.rot[ i ], self.ratio[ i ] )
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

		self.rot += self.rots
		for i in np.nonzero( ( self.rot >= 360 ) | ( self.rot < 0) )[ 0 ]:
			self.rot[ i ] = 360 + self.rot[ i ] if self.rot[ i ] < 0 else self.rot[ i ] - 360

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
