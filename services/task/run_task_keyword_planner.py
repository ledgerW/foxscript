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
from utils.FoxLLM import *


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
    topics_df = pd.read_csv(topics_path)
    volume_col = 'Search Volume' if 'Search Volume' in topics_df.columns else 'Volume'

    try:
        topics_df = topics_df\
            .assign(Volume=lambda df: df[volume_col].apply(lambda x: int(x.replace(',',''))))\
            [['Keyword', 'Volume']]
    except:
        topics_df = topics_df\
            .assign(Volume=lambda df: df[volume_col].apply(lambda x: int(x)))\
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
        .assign(AlreadyRanks=lambda df: df.AlreadyRanks.apply(lambda x: round(sum(x)/len(x), 2)))\
        .sort_values(by=['Score', 'Volume'], ascending=False)

    # Upload Merged Clusters to Bubble
    local_final_path = os.path.join(LAMBDA_DATA_DIR, f'{domain_name}_clustered_ecs.csv')

    final_df.to_csv(local_final_path, index=False)

    return local_final_path


def get_topic_category(urls: str, categories: str):
    llm = FoxLLM(az_openai_kwargs, openai_kwargs, 'gpt-4', temp=0.1)

    prompt = f"""Please assign a general category label for the urls below.
    URLs:
    {urls}


    If there is a suitable category label from the options below, then please use that.
    But if none of them fit, then create a new cateogory label. The category label does
    not have to be perfect.

    Options:
    {categories}

    The category label should be pretty simple, like Audio, Math, Home Financing, etc...

    Return only the category label.

    Category Label:
    """

    res = llm.llm.invoke(prompt)
    
    return res.content


def get_spoke_name(url: str, options: str, hub_names:str):
    llm = FoxLLM(az_openai_kwargs, openai_kwargs, 'gpt-4', temp=0.1)

    prompt = f"""Please assign a user-friendly label to the url below
    based on the text in the url. Assume the url text is representative of
    the content on its page and this label will let people know what the url
    is about. But do not make the label too specific. Do not include company
    names, for example. The label should be more specifc than a broad category,
    but still general enough to act as a sub-category, and not so specific that
    it only describes the url.
    
    URL: {url}

    The label must NOT be any of the following:
    {hub_names}

    Return only the user-friendly label.

    User-Friendly Label:
    """

    res = llm.llm.invoke(prompt)
    
    return res.content


def make_keyword_planner_doc(clustered_ecs_path, domain_name):
    raw_keyword_plan_df = pd.read_csv(clustered_ecs_path)

    subtopics_df = raw_keyword_plan_df\
        .dropna()\
        .rename(columns={'Topic': 'SubTopic', 'ClosestURL': 'MainTopic'})
    
    main_topics = raw_keyword_plan_df\
        .dropna()\
        .rename(columns={'Topic': 'SubTopic', 'ClosestURL': 'MainTopic'})\
        .groupby(['MainTopic'], as_index=False)\
        .agg(TopicVolume=('Volume', 'sum'))\
        .sort_values(by=['TopicVolume'], ascending=False)\
        .reset_index(drop=True)

    main_topics = main_topics\
        .merge(subtopics_df, how='left', on='MainTopic')\
        .groupby(['MainTopic', 'TopicVolume', 'SubTopic', 'Volume', 'Score', 'KeywordCount', 'Keywords', 'Links', 'AlreadyRanks'])\
        .agg(TopicVolumeDummy=('TopicVolume', 'first'))\
        .drop(columns=['TopicVolumeDummy'])\
        .sort_values(by=['TopicVolume', 'Volume'], ascending=False)
    
    main_topics_dict_list = main_topics\
        .reset_index()\
        .groupby('MainTopic', as_index=False)\
        .agg(Links=('Links', 'first'))\
        .to_dict(orient='records')
    
    category_labels = []
    for idx, topic in enumerate(main_topics_dict_list):
        topic['TopicNumber'] = idx+1
        topic['Label'] = get_topic_category(topic['Links'], str(category_labels))
        print(topic['Label'])

        main_topics_dict_list[idx] = topic

        category_labels = list(set(category_labels + [topic['Label']]))

    topic_labels_df = pd.DataFrame(main_topics_dict_list)

    keyword_plan_df = topic_labels_df[['Label', 'TopicNumber', 'MainTopic']]\
        .merge(main_topics.reset_index(), how='left', on='MainTopic')\
        .rename(columns={'Label': 'Category'})

    # Hubs - Spokes - Keywords
    spoke_names = []
    spoke_options = []
    hub_names = keyword_plan_df.Category.unique().tolist()
    for spoke_url in keyword_plan_df.MainTopic.unique():
        new_spoke_name = get_spoke_name(spoke_url, str(spoke_options), str(hub_names))
        spoke_names.append(new_spoke_name)
        spoke_options = list(set(spoke_options + [new_spoke_name]))

    spokes = {url: spoke for url, spoke in zip(keyword_plan_df.MainTopic.unique(), spoke_names)}

    keyword_plan_df['MainTopicURL'] = keyword_plan_df['MainTopic']
    keyword_plan_df['MainTopic'] = keyword_plan_df['MainTopicURL'].apply(lambda x: spokes[x])

    hub_and_spoke_df = keyword_plan_df\
        .groupby(['Category', 'TopicNumber'], as_index=False)\
        .agg(
            MainTopic=('MainTopic', 'first'),
            CategoryVolume=('TopicVolume', 'first'),
            KeywordCount=('KeywordCount', 'sum'),
            SubTopicCount=('SubTopic', 'count')
        )

    hub_and_spoke_df = hub_and_spoke_df\
        .groupby(['Category', 'MainTopic'], as_index=False)\
        .agg(
            CategoryVolume=('CategoryVolume', 'sum'),
            KeywordCount=('KeywordCount', 'sum'),
            SubTopicCount=('SubTopicCount', 'sum')
        )

    hub_df = hub_and_spoke_df\
        .groupby('Category', as_index=False)\
        .agg(
            CategoryVolume=('CategoryVolume', 'sum'),
            TopicCount=('MainTopic', 'count'),
            SubTopicCount=('SubTopicCount', 'sum'),
            KeywordCount=('KeywordCount', 'sum')
        )\
        .sort_values(by='CategoryVolume', ascending=False)

    renames = {
        'Category': 'Hub',
        'CategoryVolume': 'HubVolume',
        'TopicCount': 'HubSpokeCount',
        'SubTopicCount': 'HubKeywordsCount',
        'KeywordCount': 'HubClusteredKeywordCount',
        'MainTopic': 'Spoke',
        'TopicVolume': 'SpokeVolume',
        'SubTopic': 'Keyword',
        'Volume': 'KeywordVolume',
        'Keywords': 'ClusteredKeywords'
    }

    col_order = [
        'Hub', 'HubVolume', 'Spoke', 'SpokeVolume', 'Keyword', 'KeywordVolume', 'ClusteredKeywordCount', 'ClusteredKeywords',
        'Links', 'Score', 'AlreadyRanks'
    ]

    spoke_totals_df = keyword_plan_df\
        .groupby('TopicNumber', as_index=False)\
        .agg(
            Category=('Category', 'first'),
            MainTopic=('MainTopic', 'first'),
            TopicVolume=('TopicVolume', 'first')
        )\
        .groupby('MainTopic', as_index=False)\
        .agg(
            Category=('Category', 'first'),
            TopicVolume=('TopicVolume', 'sum')
        )\
        .sort_values(by=['Category', 'TopicVolume'], ascending=False)\
        [['Category', 'MainTopic', 'TopicVolume']]\
        .merge(keyword_plan_df.drop(columns=['Category', 'TopicVolume']), how='left', on='MainTopic')\
        .rename(columns={'KeywordCount': 'ClusteredKeywordCount'})

    hub_spoke_keyword_df = hub_df\
        .merge(spoke_totals_df, how='left', on='Category')\
        .rename(columns=renames)\
        .sort_values(by=['HubVolume', 'SpokeVolume', 'KeywordVolume'], ascending=False)\
        .drop(columns=['MainTopicURL'])
    
    #hub_spoke_keyword_df['ClusteredKeywordCount'] = hub_spoke_keyword_df['ClusteredKeywords'].apply(lambda x: len(x.split('-')))
    
    # Upload Merged Clusters to Bubble
    local_keyword_plan_path = os.path.join(LAMBDA_DATA_DIR, f'{domain_name}_hubs_and_spokes.csv')

    hub_spoke_keyword_df.to_csv(local_keyword_plan_path, index=False)

    return local_keyword_plan_path


def text_to_wv_classname(text):
    return ''.join(e for e in text if e.isalnum() and not e.isdigit()).capitalize()




def main(task_args):
    print('task_args:')
    print(task_args)
    print('')

    email = task_args['user_email']
    top_n_ser = task_args['top_n_ser']
    ecs_concurrency = task_args['ecs_concurrency']
    ecs_job_id = task_args['ecs_job_id']

    job_body = {
        'is_running': True
    }
    res = update_bubble_object('ecs-job', ecs_job_id, job_body)

    # Get ECS Job
    ecs_job_res = get_bubble_object('ecs-job', ecs_job_id)
    ecs_job_json = ecs_job_res.json()['response']

    # Get Compnay Domain
    domain = ecs_job_json['company_domain']
    domain_name = domain.split('.')[0]

    # Make HUB Content Library
    job_name = ecs_job_json['name']
    ec_lib_name = text_to_wv_classname(job_name)
    ec_class_name, account_name = get_wv_class_name(email, ec_lib_name)
    create_library(ec_class_name)

    # Fetch Keywords Doc input file from bubble
    keywords_doc = ecs_job_json['keywords_doc']
    keyword_doc_res = get_bubble_object('ecs-doc', keywords_doc)
    keywords_doc_url = keyword_doc_res.json()['response']['url']

    keyword_doc_file_name = keywords_doc_url.split('/')[-1]
    local_keyword_doc_path = f'{LAMBDA_DATA_DIR}/{keyword_doc_file_name}'

    if 'app.foxscript.ai' in keywords_doc_url:
        get_bubble_doc(keywords_doc_url, local_keyword_doc_path)
        print("Retrieved batch doc from bubble")
    else:
        local_keyword_doc_path = keywords_doc_url
        print("Using local batch file")


    # Get Topics DF
    topics_df = pd.read_csv(local_keyword_doc_path)
    if 'Search Volume' in topics_df.columns:
        topics_df.rename(columns={'Search Volume': 'Volume'}, inplace=True)

    print(f"Topics Shape: {topics_df.shape}")
    topics = topics_df.Keyword.to_list()

    #### STEP 1: Get SERP Results
    print('BEGIN STEP 1')

    sqs = 'ecs{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
    queue = SQS(sqs)
    serp_concurrency = 50
    all_topic_urls = []
    for i in range(0, len(topics), serp_concurrency):
        topic_batch = topics[i:i + serp_concurrency]
        
        for idx, topic in enumerate(topic_batch):
            # do distributed ECS for each topic
            cloud_ecs(topic, 'SERP', email, domain, top_n_ser, urls=[], special_return='serp', sqs=sqs, invocation_type='Event') 

        # wait for and collect search results from SQS
        print(f"Waiting for items {i} through {(i + len(topic_batch))}")
        serp_batch = queue.collect(len(topic_batch), max_wait=600, self_destruct=False)
        all_topic_urls = all_topic_urls + serp_batch
        
        # Update Job Status
        job_body = {
            'serp_progress': i
        }
        res = update_bubble_object('ecs-job', ecs_job_id, job_body)

    # Update Job Status
    job_body = {
        'serp_progress': len(topics)
    }
    res = update_bubble_object('ecs-job', ecs_job_id, job_body)
    
    print(f"all_topic_urls length: {len(all_topic_urls)}")
    queue.self_destruct()

    serp_results_df = pd.DataFrame(all_topic_urls)\
        .merge(topics_df, how='left', left_on='topic', right_on='Keyword')\
        .assign(top_serp=lambda df: df.search_urls.apply(lambda x: x.split(',')[0]))
    
    local_serp_results_path = f'{LAMBDA_DATA_DIR}/serp_results.csv'
    serp_results_df.to_csv(local_serp_results_path, index=False)

    # Save to ECS-Doc Object
    serp_results_file_url = upload_bubble_file(local_serp_results_path)
    doc_body = {
        'name': 'serp_results.csv',
        'url': serp_results_file_url,
        'type': 'serp_results_doc',
        'ecs_job': ecs_job_id
    }
    res = create_bubble_object('ecs-doc', doc_body)
    serp_results_doc_id = res.json()['id']
    
    print(f'SERP RESULTS DF SHAPE FULL: {serp_results_df.shape}')
    
    #### END STEP 1


    #### BEGIN STEP 2: Get Spokes and URLs for WV Library for ECS
    print('BEGIN STEP 2')
    
    # Now Cluster Results
    job_body = {
        'has_clustering_begun': True
    }
    res = update_bubble_object('ecs-job', ecs_job_id, job_body)

    cluster = cluster_keywords(thresh=0.4)
    input = {'input': local_serp_results_path}
    cluster_path = cluster(input, keyword_col='topic', to_bubble=False)
    print(cluster_path)

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

    clusters_df = pd.read_csv(cluster_path)
    unfolded_clusters_df = unfold_keyword_clusters(clusters_df)

    spoke_df = unfolded_clusters_df\
        .merge(serp_results_df, how='left', on='Keyword')\
        .groupby('Group', as_index=False)\
            .agg(
                Topic=('topic', 'first'),
                Volume=('Volume', 'sum'),
                Keywords=('Keyword', list),
                KeywordCount=('Keyword', 'count'),
                Links=('Links', 'first'),
                TopSERP=('top_serp', 'first')
            )\
        .sort_values(by=['Volume'], ascending=False)\
        .reset_index(drop=True)\
        .groupby('TopSERP', as_index=False)\
            .agg(
                Topic=('Topic', 'first'),
                Volume=('Volume', 'sum'),
                Keywords=('Keywords', 'sum'),
                KeywordCount=('KeywordCount', 'sum'),
                Links=('Links', 'sum')
            )\
        .sort_values(by=['Volume'], ascending=False)\
        .reset_index(drop=True)

    print(spoke_df.shape)

    # Save to ECS-Doc Object
    local_spoke_df_path = f'{LAMBDA_DATA_DIR}/spoke_df.csv'
    spoke_df.to_csv(local_spoke_df_path, index=False)
    spoke_df_file_url = upload_bubble_file(local_spoke_df_path)
    doc_body = {
        'name': f'{domain_name}_spoke_df.csv',
        'url': spoke_df_file_url,
        'type': 'spoke_df_doc',
        'ecs_job': ecs_job_id
    }
    res = create_bubble_object('ecs-doc', doc_body)
    ecs_spoke_df_doc_id = res.json()['id']

    #SPOKE_URL_MAX = 100
    #SPOKE_DF_50 = int(spoke_df.shape[0] * 0.01)
    #SPOKE_URL_COUNT = min(SPOKE_DF_50, SPOKE_URL_MAX)
    #print(SPOKE_URL_COUNT)
    SPOKE_URL_COUNT = int(spoke_df.shape[0] * 0.01)
    SPOKE_URL_COUNT = max(10, SPOKE_URL_COUNT)

    volume_spoke_urls = spoke_df.head(SPOKE_URL_COUNT).TopSERP.to_list()
    random_spoke_urls = spoke_df.sample(SPOKE_URL_COUNT).TopSERP.to_list()
    spoke_urls = volume_spoke_urls + random_spoke_urls

    #### END STEP 2


    #### BEGIN STEP 3: Load HUB URLs into WV Library
    print('BEGIN STEP 3')
    print(f"{len(spoke_urls)} unique SERPS going to WV Library")

    # Scrape and load content urls to Weaviate Library
    #content_urls = ecs_job_json['content_urls'].split('\n')
    for content_url in spoke_urls:
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
        time.sleep(0.2)
    time.sleep(180)


    # STEP 4 ECS Score Keywords into HUBs
    # Process the Batch CSV
    sqs = 'ecs{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
    queue = SQS(sqs)
    all_ecs = []
    for i in range(0, len(topics), ecs_concurrency):
        topic_batch = topics[i:i + ecs_concurrency]
        
        for idx, topic in enumerate(topic_batch):
            # do distributed ECS for each topic
            topic_urls = serp_results_df.query(f'topic == "{topic}"').search_urls.values[0].split(',')
            cloud_ecs(topic, ec_lib_name, email, domain, top_n_ser, urls=topic_urls, special_return=None, sqs=sqs, invocation_type='Event') 

        # wait for and collect search results from SQS
        print(f"Waiting for items {i} through {(i + len(topic_batch))}")
        ecs_batch = queue.collect(len(topic_batch), max_wait=600, self_destruct=False)
        all_ecs = all_ecs + ecs_batch
        
        # Update Job Status
        job_body = {
            'ecs_progress': i
        }
        res = update_bubble_object('ecs-job', ecs_job_id, job_body)

    # Update Job Status
    job_body = {
        'ecs_progress': len(topics)
    }
    res = update_bubble_object('ecs-job', ecs_job_id, job_body)
    
    print(f"all_ecs length: {len(all_ecs)}")
    queue.self_destruct()

    ecs_full_df = pd.DataFrame(all_ecs)
    print(f'ECS DF SHAPE FULL: {ecs_full_df.shape}')

    # Save Full ECS CSV
    ecs_file_name = f'{domain_name}_ecs_full.csv'
    local_ecs_path = f'{LAMBDA_DATA_DIR}/{ecs_file_name}'
    print(local_ecs_path)
    ecs_full_df.to_csv(local_ecs_path, index=False)

    # Save to ECS-Doc Object
    ecs_file_url = upload_bubble_file(local_ecs_path)
    doc_body = {
        'name': ecs_file_name,
        'url': ecs_file_url,
        'type': 'raw_ecs_full_doc',
        'ecs_job': ecs_job_id
    }
    res = create_bubble_object('ecs-doc', doc_body)
    ecs_ecs_doc_id = res.json()['id']

    # Filter out ECS Scores below 0.45
    try:
        ecs_df = ecs_full_df.query('score >= 0.5')
        print(f'ECS DF SHAPE AFTER SCORE FILTER: {ecs_df.shape}')

        domain_name = domain.split('.')[0]
        ecs_file_name = f'{domain_name}_ecs_filtered.csv'
        local_ecs_path = f'{LAMBDA_DATA_DIR}/{ecs_file_name}'
        print(local_ecs_path)
        ecs_df.to_csv(local_ecs_path, index=False)


        # Save to ECS-Doc Object
        ecs_file_url = upload_bubble_file(local_ecs_path)
        doc_body = {
            'name': ecs_file_name,
            'url': ecs_file_url,
            'type': 'raw_ecs_filtered_doc',
            'ecs_job': ecs_job_id
        }
        res = create_bubble_object('ecs-doc', doc_body)
        ecs_ecs_doc_id = res.json()['id']
    except:
        ecs_df = ecs_full_df


    # Now Cluster Results
    #job_body = {
    #    'has_clustering_begun': True
    #}
    #res = update_bubble_object('ecs-job', ecs_job_id, job_body)

    #cluster = cluster_keywords(thresh=0.4)
    #try:
    #    input = {'input': local_ecs_path}
    #    cluster_path = cluster(input, keyword_col='topic', to_bubble=False)
    #except:
    #    ecs_file_name = f'{domain_name}_ecs_full.csv'
    #    local_ecs_path = f'{LAMBDA_DATA_DIR}/{ecs_file_name}'
    #    input = {'input': local_ecs_path}
    #    cluster_path = cluster(input, keyword_col='topic', to_bubble=False)

    # Save to ECS-Doc Object
    #cluster_file_url = upload_bubble_file(cluster_path)
    #doc_body = {
    #    'name': f'{domain_name}_clusters.csv',
    #    'url': cluster_file_url,
    #    'type': 'raw_cluster_doc',
    #    'ecs_job': ecs_job_id
    #}
    #res = create_bubble_object('ecs-doc', doc_body)
    #ecs_cluster_doc_id = res.json()['id']



    # Merge into final doc
    final_doc_path = make_final_doc(local_keyword_doc_path, local_ecs_path, cluster_path, domain_name)
    keyword_plan_path = make_keyword_planner_doc(final_doc_path, domain_name)

    # Save to ECS-Doc Object
    keyword_plan_url = upload_bubble_file(keyword_plan_path)
    doc_body = {
        'name': f'{domain_name}_keyword_plan.csv',
        'url': keyword_plan_url,
        'type': 'keyword_plan_doc',
        'ecs_job': ecs_job_id
    }
    res = create_bubble_object('ecs-doc', doc_body)
    keyword_plan_id = res.json()['id']

    # Attch output to ECS Job object
    job_body = {
        'serp_results_doc': serp_results_doc_id,
        'raw_ecs_result': ecs_ecs_doc_id,
        'raw_cluster_result': ecs_cluster_doc_id,
        'keyword_plan_doc': keyword_plan_id,
        'hub_df_doc': ecs_spoke_df_doc_id,
        'cost': keyword_doc_res.json()['response']['cost'],
        'is_complete': True,
        'is_running': False
    }
    res = update_bubble_object('ecs-job', ecs_job_id, job_body)

    # Delete Wv Library
    delete_library(ec_class_name)


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

  