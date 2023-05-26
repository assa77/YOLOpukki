###############################################################################
#
#	configs.trt.fp16.py	YOLO configuration file (YOLOv3, TensorRT FP16)
#
#
#  Portions Copyright (c) 2023 by Alexander M. Albertian, <assa@4ip.ru>.
#  All rights reserved.
###############################################################################

# YOLO options
YOLO_TYPE			= "yolov3"	# "yolov4" or "yolov3"
YOLO_TINY			= True
YOLO_FRAMEWORK			= "trt"		# "tf" or "trt"
YOLO_CHECKPOINTS		= "checkpoints"
YOLO_V3_WEIGHTS			= "yolov3.weights"
YOLO_V4_WEIGHTS			= "yolov4.weights"
YOLO_V3_TINY_WEIGHTS		= "yolov3-tiny.weights"
YOLO_V4_TINY_WEIGHTS		= "yolov4-tiny.weights"
YOLO_TRT_QUANTIZE_MODE		= "FP16"	# "INT8", "FP16", "FP32"
YOLO_CUSTOM_WEIGHTS		= True		# "checkpoints/yolov3-custom"
						# used in evaluate_mAP.py and custom model detection, if not using leave False
						# YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_CLASSES			= "coco/coco.names"
YOLO_TRAIN			= "coco/train2017.txt"
YOLO_VAL			= "coco/val2017.txt"
YOLO_IOU_LOSS_THRESH		= 0.5
YOLO_ANCHOR_PER_SCALE		= 3
YOLO_MAX_BBOX_PER_SCALE		= 100
YOLO_INPUT_SIZE			= 416

# Train options
TRAIN_SAVE_BEST_ONLY		= True		# saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT		= False		# saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES			= "mnist/mnist.names"
TRAIN_ANNOTATIONS		= "mnist/mnist_train.txt"
TRAIN_LOGDIR			= "logv3"
TRAIN_MODEL_NAME		= f"{YOLO_TYPE}-custom"
TRAIN_LOAD_IMAGES_TO_RAM	= True		# With True faster training, but need more RAM
TRAIN_BATCH_SIZE		= 4
TRAIN_INPUT_SIZE		= 416
TRAIN_DATA_AUG			= True
TRAIN_TRANSFER			= False
TRAIN_FROM_CHECKPOINT		= False		# "checkpoints/yolov3-custom"
TRAIN_LR_INIT			= 1e-4
TRAIN_LR_END			= 1e-6
TRAIN_WARMUP_EPOCHS		= 4
TRAIN_EPOCHS			= 100

# TEST options
TEST_ANNOTATIONS		= "mnist/mnist_test.txt"
TEST_BATCH_SIZE			= 4
TEST_INPUT_SIZE			= 416
TEST_DATA_AUG			= False
TEST_SCORE_THRESHOLD		= 0.3
TEST_IOU_THRESHOLD		= 0.45
