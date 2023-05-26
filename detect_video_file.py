###############################################################################
#
#	detect_video_file.py	Video file detection test using YOLO (MP)
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

#import set_working_directory
import os
import sys
from pathlib import Path
from multiprocessing import current_process
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0' if current_process( ).name == "YOLO" else '-1'
#os.environ[ 'TF_ENABLE_GPU_GARBAGE_COLLECTION' ] = 'false'
from yolov3.classes import *
#from generator_zla import generator
#from yolov3.utils import Load_Yolo_model
#from yolov3.utils import detect_realtime
from yolov3.utils import detect_video_realtime_mp

def main( ):
	video_source = ""
	realtime = True
	output_path = "output.avi"
	if len( sys.argv ) > 1 and sys.argv[ 1 ]:
		video_source = sys.argv[ 1 ]
		realtime = False
		output_path = Path( video_source ).stem + ".avi"
		print( f'Video source: "{video_source}"' )
	print( f'Output file: "{output_path}"' )

#	yolo = Load_Yolo_model( )
#	detect_realtime( yolo, None, output_path, input_size = YOLO_INPUT_SIZE, width = 640, height = 480, show = True )#, rectangle_colors = ( 255, 0, 0 ) )

	detect_video_realtime_mp( None, video_source, output_path, input_size = YOLO_INPUT_SIZE, width = 640, height = 480, show = True, realtime = realtime )#, rectangle_colors = ( 255, 0, 0 ) )

if __name__ == '__main__':
	main( )
