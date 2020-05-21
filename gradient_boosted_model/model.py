
import shutil
import numpy as np
import os
import tensorflow as tf

from google.cloud import storage
import re

import matplotlib
matplotlib.use('agg',force=True) # supress python-tk error in gcloud
from matplotlib import pyplot as plt
from matplotlib.backends import backend_agg
    
from matplotlib import figure
import pandas as pd
from sklearn.metrics import roc_curve, auc
import seaborn as sns
plt.style.use('fivethirtyeight')

from functools import wraps

tf.logging.set_verbosity(tf.logging.INFO)
tf.enable_eager_execution()

BUCKET = None  # set from task.py

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
        
        file_path = 'gs://{}/credit_default/preproc/{}*'.format(BUCKET, prefix)

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
                       PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6] 
    return feature_columns


def serve_input():
    feature_placeholders = {
        'LIMIT_BAL':tf.placeholder(shape = [None], dtype = tf.float32),
        'SEX': tf.placeholder(dtype = tf.string, shape = [None]),
        'MARRIAGE': tf.placeholder(dtype = tf.string, shape =[None]),
        'EDUCATION': tf.placeholder(dtype = tf.string, shape =[None]),
        'AGE': tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_1':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_2':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_3':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_4':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_5':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_6':tf.placeholder(dtype =tf.float32, shape =[None]),
        'BILL_AMT1':tf.placeholder(dtype =tf.float32, shape =[None]),
        'BILL_AMT2':tf.placeholder(dtype =tf.float32, shape =[None]),
        'BILL_AMT3':tf.placeholder(dtype =tf.float32, shape =[None]),
        'BILL_AMT4':tf.placeholder(dtype =tf.float32, shape =[None]),
        'BILL_AMT5':tf.placeholder(dtype =tf.float32, shape =[None]),
        'BILL_AMT6':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_AMT1':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_AMT2':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_AMT3':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_AMT4':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_AMT5':tf.placeholder(dtype =tf.float32, shape =[None]),
        'PAY_AMT6':tf.placeholder(dtype =tf.float32, shape =[None]),
        KEY_COLUMN: tf.placeholder_with_default(tf.constant(['nokey']), shape =[None])
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

### Download data from GCS
def download_from_gcs(source, destination):
    search = re.search('gs://(.*?)/(.*)', source)
    bucket_name = search.group(1)
    blob_name = search.group(2)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(destination)

### Get Predictions
def predict(estimator, output_dir):
    ### Eval dataset Predictions
    predictions = estimator.predict(
        input_fn = read_dataset('eval', tf.estimator.ModeKeys.PREDICT, 2**15)
    )

    # Extract Prediction Score and Predicted Class
    preds = list()
    for pr in list(predictions):
        preds.append([pr['logistic'][0],pr['class_ids'][0],str(pr['key'])[2:-1]])
    pred_df = pd.DataFrame(preds)
    pred_df.columns = ['pred_score','pred_label','key']
    
    ### Get True Labels
    eval_path = 'gs://{}/credit_default/preproc/{}*'.format(BUCKET, 'eval')
   
    #eval_list = tf.gfile.glob(eval_path)
    eval_list = tf.io.gfile.glob(eval_path)
    
    # Merge eval files
    df_list = []
    for file in eval_list:
        df_list.append(pd.read_csv(file))
    eval_df = pd.concat(df_list)
    
    eval_df.columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_1','PAY_2','PAY_3',
               'PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',
               'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
               'PAY_AMT5','PAY_AMT6','true_label','key']

    # Add ture labels
    result = pd.merge(pred_df, eval_df,on='key')
    
    # Delete NaN values
    result.dropna(inplace=True)    
    
    ### Save the dataframe to csv in output_dir
    result.to_csv(os.path.join(output_dir,r'pred.csv'))
    
    ### Get AUC
    Eval = estimator.evaluate(
        input_fn = read_dataset('eval', tf.estimator.ModeKeys.EVAL, 2**15)
    )
    auc = Eval['auc']
    return result, auc

    
def tfgfile_wrapper(f):
    # When an example writes to local disk, change it to write to GCS.  
    # This assumes the filename argument is called 'fname' 
    # and is passed in either as a keyword argument or as the last non-keyword argument.
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'fname' in kwargs:
            fname = kwargs.pop('fname')
        else:
            args = list(args)
            fname = args.pop(-1)
        with tf.gfile.GFile(fname, 'w') as fobj:
            kwargs['fname'] = fobj
            return_value = f(*args, **kwargs)
        return return_value

    return wrapper

@tfgfile_wrapper
### Inspect incorrectly predicted instances
def inspect(estimator, output_dir, fname):
    
    ### Predictions
    df, _ = predict(estimator, output_dir)
    
    wrong_df = df[df['pred_label']!=df['true_label']]
    
    ### Draw Diagnostic Plots
    fig = figure.Figure(figsize=(24, 24))
    canvas = backend_agg.FigureCanvasAgg(fig)
    color_array=['green','red','navy','orange','purple','steelblue']
    
    default = wrong_df[wrong_df['true_label']==1]
    non_default = wrong_df[wrong_df['true_label']==0]
    
    ### Proportion of Default vs. Not Default
    ax1 = plt.subplot2grid((4,4),(0,0), fig = fig)
    ax1.bar([0,1], [len(non_default), len(default)],color=color_array[:2])
    ax1.set_title('Default Distribution')
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['Not Default','Default'], ha='center', rotation=35)
    ax1.grid(axis='x')
    
    ### Distribution of Gender
    ax2 = plt.subplot2grid((4,4),(0,1), fig=fig)
    w2 = 0.35
    x2 = np.arange(2)
    ax2.bar(x2-w2/2, [len(non_default[non_default['SEX']=='Male']),
                        len(non_default[non_default['SEX']=='Female'])]
            , w2, label='Not Default')
    ax2.bar(x2+w2/2, [len(default[default['SEX']=='Male']),len(default[default['SEX']=='Female'])], 
            w2, label='Default')
    ax2.set_title('Gender Distribution')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['Male','Female'], ha='center', rotation=35)
    ax2.grid(axis='x')    
    ax2.legend()
    
    ### Distribution of Marriage
    ax3 = plt.subplot2grid((4,4),(0,2), fig=fig)
    w3 = 0.30
    x3 = np.arange(4)
    ax3.bar(x3-w3/2, [len(non_default[non_default['MARRIAGE']=='Unknown']),
                        len(non_default[non_default['MARRIAGE']=='Married']),
                        len(non_default[non_default['MARRIAGE']=='Single']),
                        len(non_default[non_default['MARRIAGE']=='Others'])]
            , w3, label='Not Default')
    ax3.bar(x3+w3/2, [len(default[default['MARRIAGE']=='Unknown']),
                        len(default[default['MARRIAGE']=='Married']),
                        len(default[default['MARRIAGE']=='Single']),
                        len(default[default['MARRIAGE']=='Others'])], 
            w3, label='Default')
    ax3.set_title('Marital Status')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(['Unknown','Married','Single','Others'], ha='center', rotation=35)
    ax3.grid(axis='x')    
    ax3.legend()
    
    ### Education Distribution
    ax4 = plt.subplot2grid((4,4),(0,3), fig=fig)
    w4 = 0.25
    x4 = np.arange(5)
    ax4.bar(x4-w4/2, [len(non_default[non_default['EDUCATION']=='Graduate']),
                        len(non_default[non_default['EDUCATION']=='University']),
                        len(non_default[non_default['EDUCATION']=='High-school']),
                        len(non_default[non_default['EDUCATION']=='Unknown1']),
                        len(non_default[non_default['EDUCATION']=='Others'])]
            , w4, label='Not Default')
    ax4.bar(x4+w4/2, [len(default[default['EDUCATION']=='Graduate']),
                        len(default[default['EDUCATION']=='University']),
                        len(default[default['EDUCATION']=='High-school']),
                        len(default[default['EDUCATION']=='Unknown1']),
                        len(default[default['EDUCATION']=='Others'])], 
            w4, label='Default')
    ax4.set_title('Education Status')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(['Graduate','University','High-school','Unknown1','Others'], 
                        ha='center', rotation=35)
    ax4.grid(axis='x')    
    ax4.legend()    
    
    ### Age Distribution
    ax5 = plt.subplot2grid((4,4),(1,0), colspan = 2,fig=fig)
    ax5.hist(non_default['AGE'], bins=200, color='green', density=True, label='Not Default')
    ax5.hist(default['AGE'], bins=200, color='red', density=True, label='Default')
    ax5.set_title('AGE Grouped by Default Class')
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Normalized Densities')
    ax5.legend()

    ### Credit Limit Distribution
    ax6 = plt.subplot2grid((4,4),(1,2), colspan = 2,fig=fig)
    ax6.hist(non_default['LIMIT_BAL'], bins=200, color='green', density=True, label='Not Default')
    ax6.hist(default['LIMIT_BAL'], bins=200, color='red', density=True, label='Default')
    ax6.set_title('Credit Limit Grouped by Default Class')
    ax6.set_xlabel('Credit Limit')
    ax6.set_ylabel('Normalized Densities')
    ax6.legend()
    
    ### Payment Delays
    ax7 = plt.subplot2grid((4,4),(2,0), colspan = 2,fig=fig)
    x7 = np.arange(6)
    w7 = 0.25
    ax7.bar(x7-w7/2, [non_default['PAY_1'].mean(),
                      non_default['PAY_2'].mean(),
                      non_default['PAY_3'].mean(),
                      non_default['PAY_4'].mean(),
                      non_default['PAY_5'].mean(),
                      non_default['PAY_6'].mean()]
            , w7, label='Not Default')
    ax7.bar(x7+w7/2, [default['PAY_1'].mean(),
                      default['PAY_2'].mean(),
                      default['PAY_3'].mean(),
                      default['PAY_4'].mean(),
                      default['PAY_5'].mean(),
                      default['PAY_6'].mean()], 
            w7, label='Default')
    ax7.set_title('Average Monthly Payment Delays')
    ax7.set_xlabel('')
    ax7.set_ylabel('Average Monthly Delays')
    ax7.set_xticks(x7)
    ax7.set_xticklabels(['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'], 
                        ha='center', rotation=35)
    ax7.legend()
    
    ### Monthly Bill Amounts
    ax8 = plt.subplot2grid((4,4),(2,2), colspan = 2,fig=fig)
    x8 = np.arange(6)
    labels = ['Not Default', 'Default']
    df_BILL = df.melt(id_vars='true_label', 
                      value_vars=['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'])

    df_BILL.loc[(df_BILL['true_label']==0),'true_label'] = 'Not Default'
    df_BILL.loc[(df_BILL['true_label']==1),'true_label'] = 'Default'
    sns.boxplot(x="variable", y="value", hue = "true_label", 
                hue_order=labels, data=df_BILL, linewidth=2,ax = ax8)
    ax8.set_xlabel('')
    ax8.set_xticks(x8)
    ax8.set_xticklabels(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'], 
                        ha='center')
    ax8.set_xlabel('')
    ax8.set_ylabel('Bill Amounts')
    ax8.set_title('Monthly Bill Amounts')
    ax8.legend()
    
    ### Monthly Bill Payments
    ax9 = plt.subplot2grid((4,4),(3,1), colspan = 2,fig=fig)
    x9 = np.arange(6)
    df_PAY = df.melt(id_vars='true_label', 
                      value_vars=['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'])

    sns.boxplot(x="variable", y="value", hue = "true_label", data=df_PAY, linewidth=2,ax = ax9)
    ax9.set_xlabel('')
    ax9.set_xticks(x9)
    ax9.set_xticklabels(['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], 
                        ha='center')
    ax9.set_xlabel('')
    ax9.set_yscale('log')
    ax9.set_ylabel('Bill Payment Amounts (log scale)')
    ax9.set_title('Monthly Bill Payment Amounts')
    ax9.legend('')
    
    ### Save Figure
    fig.tight_layout()
    canvas.print_figure(fname, format='png')
    
    
@tfgfile_wrapper
### Visualize Confusion Matrix and ROC Curve
def visualize(estimator, output_dir, fname):
    
    fig = figure.Figure(figsize=(12, 6))
    canvas = backend_agg.FigureCanvasAgg(fig)
    
    pred_df, auc = predict(estimator, output_dir)
    
    ### Confusion Matrix Visualization
    cm = pd.crosstab(pred_df['true_label'].values, pred_df['pred_label'].values,
                     rownames=['Actual'], colnames=['Predicted'])
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

   # fig, ax= plt.subplots(figsize=(12, 6), dpi=150, nrows = 1, ncols=2)
    ax1 = fig.add_subplot(1, 2, 1)
    cmap = sns.cubehelix_palette(8)
    sns.heatmap(cm, 
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap=cmap)
    
    ax1.set_title('Confusion Matrix', fontsize=14)
    
    ### ROC Curve
    fpr, tpr, threshold = roc_curve(pred_df['true_label'], pred_df['pred_score'])
    roc_auc = auc #auc(fpr, tpr)
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(fpr, tpr, color='darkorange',
               lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver operating characteristic example')
    ax2.legend(loc="lower right")
    
    fig.tight_layout()
    
    ### Save Figure
    canvas.print_figure(fname, format='png')
      
    
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

    ### Create visualizations
    cm_roc = os.path.join(output_dir,'cm_roc.png')
    visualize(estimator, output_dir, fname=cm_roc)
    
    insp = os.path.join(output_dir,'insp.png')
    inspect(estimator, output_dir, fname=insp)
    
    ### Download the figure from GCS
    if output_dir.startswith('gs://'):
        tf.compat.v1.logging.info('Downloading cm_roc from GCS')
        download_from_gcs(cm_roc, destination='cm_roc.png')
        download_from_gcs(insp, destination='insp.png')
