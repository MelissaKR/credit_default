
import shutil
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
tf.enable_eager_execution()

BUCKET = None  # set from task.py
PATTERN = 'of' # gets all files

# Determine CSV, label, and key columns
CSV_COLUMNS = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_1','PAY_2','PAY_3',\
               'PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',\
               'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',\
               'PAY_AMT5','PAY_AMT6','default_payment','key']
LABEL_COLUMN = 'default_payment'
KEY_COLUMN = 'key'

# Set default values for each CSV column
DEFAULTS = [[0.0],['null'],['null'],['null'],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],
            [0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0],['nokey']]

# Define some default hyperparameters
EVAL_STEPS = None


### Read dataset from GCS
def read_dataset(prefix, mode, batch_size):
    def _input_fn():
        def decode_csv(value_column):
            ### Convert CSV records to tensors:
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label
        
        file_path = 'gs://{}/credit_default/preproc/{}*{}*'.format(BUCKET, prefix, PATTERN)

        file_list = tf.io.gfile.glob(file_path)
        
        ### Map csv files to tensor
        dataset = (tf.data.TextLineDataset(file_list).map(decode_csv))  
      
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None 
            ### Shuffle the datasets for each epoch of Gradient descent
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 
 
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

### Define Feature Columns
def get_features():
    
    feature_columns = []
    
    ### Categorical Features:
    def cat_column(feature_name, vocab):
        return tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab))
    
    SEX = cat_column('SEX', ['Male', 'Female'])
    MARRIAGE = cat_column('MARRIAGE', ['Unknown','Married','Single','Others'])
    EDUCATION = cat_column('EDUCATION', 
                        ['Graduate','University','High-school','Others','Unknown1','Unknown2'])
    

    ### Numeric Columns:
    NUMERIC_FEATURES = ['LIMIT_BAL','AGE','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
                       'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                       'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    
    LIMIT_BAL,AGE,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,\
    BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,\
    PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6 = \
    [tf.feature_column.numeric_column(feature) for feature in NUMERIC_FEATURES]

        
    feature_columns = [LIMIT_BAL,SEX, EDUCATION,AGE,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,
                       BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,
                       PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6] #, embed]
    return feature_columns


def serve_input():
    feature_placeholders = {
        'LIMIT_BAL':tf.placeholder(tf.float32, [None]),
        'SEX': tf.placeholder(tf.string, [None]),
        'MARRIAGE': tf.placeholder(tf.string, [None]),
        'EDUCATION': tf.placeholder(tf.string, [None]),
        'AGE': tf.placeholder(tf.float32, [None]),
        'PAY_1':tf.placeholder(tf.float32, [None]),
        'PAY_2':tf.placeholder(tf.float32, [None]),
        'PAY_3':tf.placeholder(tf.float32, [None]),
        'PAY_4':tf.placeholder(tf.float32, [None]),
        'PAY_5':tf.placeholder(tf.float32, [None]),
        'PAY_6':tf.placeholder(tf.float32, [None]),
        'BILL_AMT1':tf.placeholder(tf.float32, [None]),
        'BILL_AMT2':tf.placeholder(tf.float32, [None]),
        'BILL_AMT3':tf.placeholder(tf.float32, [None]),
        'BILL_AMT4':tf.placeholder(tf.float32, [None]),
        'BILL_AMT5':tf.placeholder(tf.float32, [None]),
        'BILL_AMT6':tf.placeholder(tf.float32, [None]),
        'PAY_AMT1':tf.placeholder(tf.float32, [None]),
        'PAY_AMT2':tf.placeholder(tf.float32, [None]),
        'PAY_AMT3':tf.placeholder(tf.float32, [None]),
        'PAY_AMT4':tf.placeholder(tf.float32, [None]),
        'PAY_AMT5':tf.placeholder(tf.float32, [None]),
        'PAY_AMT6':tf.placeholder(tf.float32, [None]),
        KEY_COLUMN: tf.placeholder_with_default(tf.constant(['nokey']), [None])
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

### Add evaluation metrics
def my_rmse(labels, predictions):
    pred_values = predictions['logistic']
    return {'auc': tf.metrics.auc(labels, pred_values, curve='ROC',
                                  summation_method='careful_interpolation'),
           'precision':tf.metrics.precision(labels, pred_values),
            'recall':tf.metrics.recall(labels, pred_values)}

### Train and Evaluate Model
def train_eval(output_dir, hparams):
    tf.summary.FileWriterCache.clear() 

    EVAL_INTERVAL = 60 # in seconds
    
    ### Get feature columns
    feature_columns = get_features()
    
    ### Specify the estimator configuration: how often to write checkpoints and how many to keep 
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                        model_dir = output_dir)
    
    ### Gradient Boosted Tree Classifier
    estimator = tf.estimator.BoostedTreesClassifier(
        model_dir = output_dir,
        feature_columns= feature_columns,
        n_trees = hparams['n_trees'],
        max_depth = hparams['max_depth'],
        n_batches_per_layer = hparams['n_batches_per_layer'],
        learning_rate = hparams['learning_rate'],
        l1_regularization = hparams['l1_regularization'],
        l2_regularization = hparams['l2_regularization'],
        center_bias=True,
        config = run_config)


    ### Add evaluating metric
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    
    ### Forward features to predictions dictionary:
    estimator = tf.contrib.estimator.forward_features(estimator, KEY_COLUMN)
    
    ### Configuration for the "train" part for the train_and_evaluate call:
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset('train', tf.estimator.ModeKeys.TRAIN, hparams['batch_size']),
        max_steps = hparams['train_examples']*1000/hparams['batch_size'])
    
    ### Export the serving graph and checkpoints for use in 
    exporter = tf.estimator.LatestExporter('exporter', serve_input, exports_to_keep=None) # disable garbage collection

    ### Configuration for the "eval" part for the train_and_evaluate call:
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset('eval', tf.estimator.ModeKeys.EVAL, 2**15), 
        steps = EVAL_STEPS,
        start_delay_secs = 60, 
        throttle_secs = EVAL_INTERVAL, 
        exporters = exporter)

    ### train and evaluate:
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
