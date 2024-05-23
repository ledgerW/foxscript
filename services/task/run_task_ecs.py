import sys
sys.path.append('..')

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import boto3
import argparse
from datetime import datetime
import time
import json
import pandas as pd
from utils.bubble import (
    create_bubble_object,
    get_bubble_object,
    update_bubble_object,
    get_bubble_doc,
    delete_bubble_object,
    upload_bubble_file
)
from utils.cloud_funcs import cloud_ecs
from utils.general import SQS
from utils.response_lib import *
#from utils.google import get_creds, upload_to_google_drive
from utils.weaviate_utils import wv_client, get_wv_class_name, create_library, delete_library
from utils.Steps import cluster_keywords
from utils.workflow_utils import unfold_keyword_clusters


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')

LAMBDA_DATA_DIR = '.'


def make_final_doc(topics_path, ecs_path, clusters_path, domain_name):
    # Get Keywords df
    topics_df = pd.read_csv(topics_path)\
        .assign(Volume=lambda df: df['Search Volume'].apply(lambda x: int(x)))\
        [['Keyword', 'Volume']]

    # Get Clusters df
    #cluster_local_file_name = os.path.join(LAMBDA_DATA_DIR, 'clusters.csv')
    #get_bubble_doc(clusters_url, cluster_local_file_name)
    clusters_df = pd.read_csv(clusters_path)

    # Unfold the Clusters DF
    unfolded_clusters_df = unfold_keyword_clusters(clusters_df)

    # Merge Keyword Volumes onto Unfolded Clusters and Group
    ecs_df = pd.read_csv(ecs_path)
    final_df = unfolded_clusters_df\
        .merge(topics_df, how='left', on='Keyword')\
        .merge(ecs_df, how='left', left_on='Keyword', right_on='topic')\
        .drop_duplicates(subset=['Keyword'])\
        .groupby('Group', as_index=False)\
        .agg(
            Topic=('topic', 'first'),
            ClosestURL=('url', 'first'),
            Score=('score', 'mean'),
            AlreadyRanks=('already_ranks', list),
            Volume=('Volume', 'sum'),
            Keywords=('Keyword', list),
            KeywordCount=('Keyword', 'count'),
            Links=('Links', 'first')
        )\
        .assign(Keywords=lambda df: df.Keywords.apply(lambda x: ' - '.join(x)))\
        .assign(AlreadyRanks=lambda df: df.AlreadyRanks.apply(lambda x: round(sum(x)/len(x), 2)))

    # Upload Merged Clusters to Bubble
    local_final_path = os.path.join(LAMBDA_DATA_DIR, f'{domain_name}_clustered_ecs.csv')

    final_df.to_csv(local_final_path, index=False)

    return local_final_path


def text_to_wv_classname(text):
    return ''.join(e for e in text if e.isalnum() and not e.isdigit()).capitalize()



def main(task_args):
    print('task_args:')
    print(task_args)
    print('')

    email = task_args['user_email']
    top_n_ser = task_args['top_n_ser']
    ecs_concurrency = task_args['ecs_concurrency']    # 0.1, 0.5, 1, etc...
    ecs_job_id = task_args['ecs_job_id']

    # Get ECS Job
    ecs_job_res = get_bubble_object('ecs-job', ecs_job_id)
    ecs_job_json = ecs_job_res['response']

    # Get Compnay Domain
    domain = ecs_job_json['company_domain']

    # Fetch Keywords Doc input file from bubble
    keywords_doc = ecs_job_json['keywords_doc']
    keyword_doc_res = get_bubble_object('ecs-doc', keywords_doc)
    keywords_doc_url = keyword_doc_res.json()['response']['url']

    batch_input_file_name = keywords_doc_url.split('/')[-1]
    local_batch_path = f'{LAMBDA_DATA_DIR}/{batch_input_file_name}'

    if 'app.foxscript.ai' in keywords_doc_url:
        get_bubble_doc(keywords_doc_url, local_batch_path)
        print("Retrieved batch doc from bubble")
    else:
        local_batch_path = keywords_doc_url
        print("Using local batch file")


    # Make Existing Content Library
    job_name = ecs_job_json['name']
    ec_lib_name = text_to_wv_classname(job_name)
    ec_class_name, account_name = get_wv_class_name(email, ec_lib_name)
    create_library(ec_class_name)

    # Scrape and load content urls to Weaviate Library
    for content_url in ecs_job_json['content_urls']:
        out_body = {
            'email': email,
            'name': ec_lib_name,
            'doc_url': content_url
        }

        _ = lambda_client.invoke(
            FunctionName=f'foxscript-api-{STAGE}-upload_to_s3_cloud',
            InvocationType='Event',
            Payload=json.dumps({"body": out_body})
        )

    # Get Topics
    topics_df = pd.read_csv(local_batch_path)
    print(f"Topics Shape: {topics_df.shape}")
    topics = [t.split(' - ')[0] for t in topics_df.Keyword]
    topics = [t for t in topics if t]
    
    # Process the Batch CSV
    sqs = 'ecs{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
    queue = SQS(sqs)
    serper_api = 'serper'
    all_ecs = []
    for i in range(0, len(topics), ecs_concurrency):
        topic_batch = topics[i:i + ecs_concurrency]
        
        for idx, topic in enumerate(topic_batch):
            # do distributed ECS for each topic
            cloud_ecs(topic, ec_lib_name, email, domain, top_n_ser, serper_api, sqs=sqs, invocation_type='Event') 

        # wait for and collect search results from SQS
        print(f"Waiting for items {i} through {(i + len(topic_batch))}")
        ecs_batch = queue.collect(len(topic_batch), max_wait=600, self_destruct=False)
        all_ecs = all_ecs + ecs_batch
        
        # Update Job Status
        job_body = {
            'ecs_progress': i
        }
        res = update_bubble_object('ecs-job', ecs_job_id, job_body)
    
    print(f"all_ecs length: {len(all_ecs)}")
    queue.self_destruct()

    ecs_df = pd.DataFrame(all_ecs)
    print(f'ECS DF SHAPE: {ecs_df.shape}')

    domain_name = domain.split('.')[0]
    ecs_file_name = f'{domain_name}_ecs.csv'
    local_ecs_path = f'{LAMBDA_DATA_DIR}/{ecs_file_name}'
    print(local_ecs_path)
    ecs_df.to_csv(local_ecs_path, index=False)

    # Save to ECS-Doc Object
    ecs_file_url = upload_bubble_file(local_ecs_path)
    doc_body = {
        'name': ecs_file_name,
        'url': ecs_file_url,
        'type': 'raw_ecs_doc',
        'ecs_job': ecs_job_id
    }
    res = create_bubble_object('ecs-doc', doc_body)
    ecs_ecs_doc_id = res.json()['id']


    # Now Cluster Results
    cluster = cluster_keywords()
    input = {'input': local_ecs_path}
    cluster_path = cluster(input, keyword_col='topic', to_bubble=False)

    # Save to ECS-Doc Object
    cluster_file_url = upload_bubble_file(cluster_path)
    doc_body = {
        'name': f'{domain_name}_clusters.csv',
        'url': cluster_file_url,
        'type': 'raw_cluster_doc',
        'ecs_job': ecs_job_id
    }
    res = create_bubble_object('ecs-doc', doc_body)
    ecs_cluster_doc_id = res.json()['id']

    # Merge into final doc
    final_doc_path = make_final_doc(local_batch_path, local_ecs_path, cluster_path, domain_name)

    # Save to ECS-Doc Object
    final_doc_url = upload_bubble_file(final_doc_path)
    doc_body = {
        'name': f'{domain_name}_ecs_clusters.csv',
        'url': final_doc_url,
        'type': 'ecs_cluster_doc',
        'ecs_job': ecs_job_id
    }
    res = create_bubble_object('ecs-doc', doc_body)
    final_doc_id = res.json()['id']

    # Attch output to ECS Job object
    job_body = {
        'raw_ecs_result': ecs_ecs_doc_id,
        'raw_cluster_result': ecs_cluster_doc_id,
        'ecs_cluster_result': final_doc_id
    }
    res = update_bubble_object('ecs-job', ecs_job_id, job_body)
    ecs_cluster_doc_id = res.json()['id']


    # Send to Google Drive
    #res = get_bubble_object('user', user_id)
    #goog_token = res.json()['response']['DriveAccessToken']
    #goog_refresh_token = res.json()['response']['DriveRefreshToken']
    
    #creds = get_creds(goog_token, goog_refresh_token)

    #file_id = upload_to_google_drive(
    #    f'{domain_name}_ecs',
    #    'csv',
    #    content=None,
    #    path=local_ecs_path,
    #    folder_id=drive_folder,
    #    creds=creds
    #)
    #print(file_id)

    #file_id = upload_to_google_drive(
    #    f'{domain_name}_clusters',
    #    'csv',
    #    content=None,
    #    path=cluster_path,
    #    folder_id=drive_folder,
    #    creds=creds
    #)
    #print(file_id)
        
    # send result to Bubble Document
    #body = {}
    #res = create_bubble_object('document', body)
    #new_doc_id = res.json()['id']

    # add new doc to project
    #res = get_bubble_object('project', project_id)
    #try:
    #    project_docs = res.json()['response']['documents']
    #except:
    #    project_docs = []

    #_ = update_bubble_object('project', project_id, {'documents': project_docs+[new_doc_id]})




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_args', default=None, type=str)
    args, _ = parser.parse_known_args()
    print(args.task_args)

    task_args = json.loads(args.task_args)

    start = datetime.now()
    print(start)
    
    main(task_args)
    
    print(datetime.now() - start)

  