###############################################################################
#
#	detect_video_mp.py	Real-time video detection test using YOLO (MP)
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

#import set_working_directory
import os
from multiprocessing import current_process
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0' if current_process( ).name == "YOLO" else '-1'
#os.environ[ 'TF_ENABLE_GPU_GARBAGE_COLLECTION' ] = 'false'
from yolov3.classes import *
from generator_zla import generator
#from yolov3.utils import Load_Yolo_model
#from yolov3.utils import detect_realtime
from yolov3.utils import detect_video_realtime_mp

def main( ):
#	yolo = Load_Yolo_model( )
#	detect_realtime( yolo, generator, "output.avi", input_size = YOLO_INPUT_SIZE, width = 640, height = 480, show = True )#, rectangle_colors = ( 255, 0, 0 ) )

#	detect_video_realtime_mp( generator, "", "output.avi", input_size = YOLO_INPUT_SIZE, width = 640, height = 480, show = True, realtime = True )#, rectangle_colors = ( 255, 0, 0 ) )
	detect_video_realtime_mp( generator, "", "output.avi", input_size = YOLO_INPUT_SIZE, width = 640, height = 480, show = True, realtime = False )#, rectangle_colors = ( 255, 0, 0 ) )

if __name__ == '__main__':
	main( )
