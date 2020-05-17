
from google.cloud import bigquery
import apache_beam as beam
from apache_beam.options import pipeline_options
print(beam.__version__)
import datetime  

import warnings
warnings.filterwarnings('ignore')
  
def to_csv(row):
    import hashlib
    ### Feature Columns
    COLUMNS = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_1','PAY_2','PAY_3',\
                'PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',\
               'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',\
               'PAY_AMT5','PAY_AMT6','default_payment']
    
    ### Replace numbers in categorical features to text labels
    row['SEX'] = ['Male', 'Female'][row['SEX'] - 1]
    row['MARRIAGE'] = ['Unknown','Married','Single','Others'][row['MARRIAGE']]
    
    # If EDUCATION==0, place it in the "Unknown2" category
    if row['EDUCATION']==0:
        row['EDUCATION'] = 5
    
    row['EDUCATION'] = ['Graduate','University','High-school','Others','Unknown1','Unknown2'][row['EDUCATION'] - 1]
    
    for pay in ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']:
        if row[pay] < 0:
            row[pay] = 0
    
    data = ','.join([str(row[k]) if k in row else 'None' for k in COLUMNS])
    
    ### Using hashids for each row as keys
    key = hashlib.sha224(data.encode('utf-8')).hexdigest()  
    yield str('{},{}'.format(data, key))


def preprocess_data(test_mode):
    import shutil, os, subprocess
    
    ### Saving the job
    job_name = 'preprocess-credit-features' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    if test_mode:
        ### If running in test mode, save the job locally
        print('Launching job in test mode:')
        OUTPUT_DIR = './preproc'
        
        # delete output directory if it exists
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        
        # create the directory
        os.makedirs(OUTPUT_DIR)
    else:
        ### If launching a Dataflow job, save the job on Google Cloud Storage (GCS)
        print('Launching Dataflow job {}:'.format(job_name))
        OUTPUT_DIR = 'gs://{0}/credit_default/preproc/'.format(BUCKET)
        try:
            subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
        except:
            pass
    
    
    ### Let's define our own Apache Beam Options:
    options = {
        'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
        'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
        'job_name': job_name,
        'region': REGION,
        'project': PROJECT,
        'max_num_workers': 6,
        'setup_file':'directory/to/setup.py'  # change this to the directory of the setup file
    }
   
    opts = pipeline_options.PipelineOptions(flags = [], **options)
    
    ### Choose the runner
    if test_mode:
        ### local mode
        RUNNER = 'DirectRunner'
    else:
        ### Dataflow
        RUNNER = 'DataflowRunner'
        
    p = beam.Pipeline(RUNNER, options = opts)
    
    ### Let's create the Train and Eval Datasets:
    query = """
        SELECT 
            ABS(FARM_FINGERPRINT(CAST(ID AS STRING))) AS hashid,
            LIMIT_BAL,
            SEX,
            EDUCATION,
            MARRIAGE,
            AGE,
            PAY_0 AS PAY_1,
            PAY_2,
            PAY_3,
            PAY_4,
            PAY_5,
            PAY_6,
            CAST(BILL_AMT1 AS FLOAT64) AS BILL_AMT1,
            CAST(BILL_AMT2 AS FLOAT64) AS BILL_AMT2,
            CAST(BILL_AMT3 AS FLOAT64) AS BILL_AMT3,
            CAST(BILL_AMT4 AS FLOAT64) AS BILL_AMT4,
            CAST(BILL_AMT5 AS FLOAT64) AS BILL_AMT5,
            CAST(BILL_AMT6 AS FLOAT64) AS BILL_AMT6,
            CAST(PAY_AMT1 AS FLOAT64) AS PAY_AMT1,
            CAST(PAY_AMT2 AS FLOAT64) AS PAY_AMT2,
            CAST(PAY_AMT3 AS FLOAT64) AS PAY_AMT3,
            CAST(PAY_AMT4 AS FLOAT64) AS PAY_AMT4,
            CAST(PAY_AMT5 AS FLOAT64) AS PAY_AMT5,
            CAST(PAY_AMT6 AS FLOAT64) AS PAY_AMT6,
            CAST(default_payment_next_month AS INT64) AS default_payment
        FROM
            `credit-default-277316.credit_default.credit_default`
        """

    if test_mode:
        query = query + ' LIMIT 100' 

    for step in ['train', 'eval']:
        if step == 'train':
            selquery = 'SELECT * FROM ({}) WHERE MOD(ABS(hashid),5) < 4'.format(query)
        else:
            selquery = 'SELECT * FROM ({}) WHERE MOD(ABS(hashid),5) = 4'.format(query)

        (p 
         | '{}_read'.format(step) >> beam.io.Read(beam.io.BigQuerySource(query = selquery, 
                                                                     use_standard_sql = True))
         | '{}_csv'.format(step) >> beam.FlatMap(to_csv)
         | '{}_out'.format(step) >> beam.io.Write(beam.io.WriteToText(
             os.path.join(OUTPUT_DIR, '{}.csv'.format(step))))
        )

    job = p.run()
    
    
    if test_mode:
        job.wait_until_finish()
        print("Done!")
