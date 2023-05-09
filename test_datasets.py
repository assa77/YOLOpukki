###############################################################################
#
#	test_datasets.py	Datasets test script
#
#
#  Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

#import set_working_directory
from yolov3.classes import *
from yolov3.dataset import Dataset

trainset = Dataset( 'train' )
print( f"Train dataset size: {trainset.num_samples}" )
testset = Dataset( 'test' )
print( f"Test dataset size: {testset.num_samples}" )
