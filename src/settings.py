"""
Settings and constants for training and deploying dusty solar panel prediction.
"""

# HYPER-PARAMETERS (training)
BATCH_SIZE = 32
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

RAND_ROTATION = 0.45
RAND_ZOOM = -0.1

VALIDATION_SPLIT =  0.2