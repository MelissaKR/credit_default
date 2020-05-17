
import argparse
import json
import os

from . import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help = 'Path to data on Google Cloud Storage. Default://BUCKET/credit_default/preproc/',
        required = True
    )
    parser.add_argument(
        '--project',
        help = 'Project ID for the current project',
        required = True
    )
    parser.add_argument(
        '--region',
        help = 'Region specified for the project',
        required = True
    )
    parser.add_argument(
        '--mode',
        type=bool,
        default=False
    )

    ### parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    ### assigning the arguments to the model variables
    preprocess.BUCKET    = arguments.pop('bucket')
    preprocess.PROJECT   = arguments.pop('project')   
    preprocess.REGION    = arguments.pop('region')
    MODE       = arguments.pop('mode')


    # Run the preprocessing job
    preprocess.preprocess_data(test_mode = MODE)
    
