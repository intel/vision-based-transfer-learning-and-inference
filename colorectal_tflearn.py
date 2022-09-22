import os
import tensorflow as tf
import argparse
from distutils.util import strtobool

from common_utils import TransferLearning
from common_utils import setting_precision

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", help="Specify the precision as FP32 or BF16",  type=str)
    parser.add_argument("--platform", help="Specify the platform as SPR or anything else",  type=str)
    parser.add_argument("--inference", help="if only inference required",  type=strtobool, default=False)
    parser.add_argument("--cp", help="Checkpoint file to be used",  type=str)
    parser.add_argument("--OUTPUT_DIR", help="Output Directory where results gets saved",  type=str)
    parser.add_argument("--DATASET_DIR", help="Dataset Directory",  type=str)
    parser.add_argument("--BATCH_SIZE", help="Batch Size",  type=int)
    parser.add_argument("--NUM_EPOCHS", help="Number of epochs for training",  type=int)
    inf_args = parser.parse_args()
    #Setting Precison
    setting_precision(inf_args)
    
    colorectal = TransferLearning(inf_args)
    colorectal.load_dataset_from_tfds()
    print("Total classes = ", colorectal.get_total_classes())
    colorectal.normalize()
    colorectal.apply_augmentation()
    if (inf_args.inference != True):
        colorectal.make_model()
        colorectal.train_model(inf_args.NUM_EPOCHS)
        colorectal.evaluate(colorectal.base_log_dir)
    else:
        colorectal.evaluate(inf_args.cp)