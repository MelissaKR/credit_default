
import argparse
import json
import os

from . import model

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help = 'Path to data on Google Cloud Storage. Default://BUCKET/credit-card/preproc/',
        required = True
    )
    parser.add_argument(
        '--output_dir',
        help = 'Location to write checkpoints and export models on Google Cloud Storage',
        required = True
    )
    parser.add_argument(
        '--batch_size',
        help = 'Size of batches of to compute gradients over in each epoch.',
        type = int,
        default = 256
    )
    parser.add_argument(
        '--job-dir',
        help = 'This model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    parser.add_argument(
        '--n_trees',
        help = 'Number of trees in the Ensemble Tree',
        type = int,
        default=50
    )
    parser.add_argument(
        '--max_depth',
        help = 'Maximum depth of each tree in the ensemble',
        type = int,
        default = 3
    )
    parser.add_argument(
        '--train_examples',
        help = 'Number of examples (in thousands) to run the training job over. This determines the number of epochs.',
        type = int,
        default = 300
    )    
    parser.add_argument(
        '--n_batches_per_layer',
        help = 'Number of batches to collect statistics per layer. Total number of batches is total number of data divided by batch size.',
        type = int,
        default = 1
    )
    parser.add_argument(
        '--learning_rate',
        help = 'Shrinkage parameter to be used when a tree added to the model',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--l1_regularization',
        help = 'Regularization multiplier applied to the absolute weights of the tree leafs',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--l2_regularization',
        help = 'Regularization multiplier applied to the square weights of the tree leafs',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--pattern',
        help = 'Specify a pattern that has to be in input files.',
        default = 'of'
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Positive number of steps for which to evaluate model. Default is None, meaning evaluate until input_fn raises an end-of-input exception',
        type = int,       
        default = None
    )
        
    ### parse all arguments
    args = parser.parse_args()
    hparams = args.__dict__
    output_dir = hparams.pop('output_dir')

    ### unused args 
    hparams.pop('job_dir', None)
    model.BUCKET  = hparams.pop('bucket')
    model.PATTERN = hparams.pop('pattern')
    
    ### In the case of hyperparameter tuning, adds trial_id to the output path:
    output_dir = os.path.join(output_dir,
        json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', ''))

    # Run the training job
    model.train_eval(output_dir, hparams)
