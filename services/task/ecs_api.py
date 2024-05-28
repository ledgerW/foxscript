import sys
sys.path.append('..')

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import json
import time
import pathlib
from datetime import datetime
import requests
import boto3
import pandas as pd
from unstructured.partition.html import partition_html
from unstructured.staging.base import convert_to_dict
from bs4 import BeautifulSoup
from utils.general import SQS
from utils.response_lib import *
from utils.weaviate_utils import wv_client, get_wv_class_name
from utils.content import handle_pdf
from utils.bubble import (
    create_bubble_object,
    get_bubble_object,
    update_bubble_object,
    get_bubble_doc,
    upload_bubble_file
)
from utils.weaviate_utils import wv_client, get_wv_class_name, create_library, delete_library
from utils.Steps import cluster_keywords
from utils.workflow_utils import unfold_keyword_clusters
from utils.cloud_funcs import cloud_ecs


from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')
SCRAPER_API_KEY = os.getenv('SCRAPER_API_KEY')

if os.getenv('IS_OFFLINE'):
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')
   LAMBDA_DATA_DIR = '/tmp'

embedder = OpenAIEmbeddings(model="text-embedding-3-large")


def make_final_doc(topics_path, ecs_path, clusters_path, domain_name):
    # Get Keywords df
    try:
        topics_df = pd.read_csv(topics_path)\
            .assign(Volume=lambda df: df['Search Volume'].apply(lambda x: int(x)))\
            [['Keyword', 'Volume']]
    except:
        topics_df = pd.read_csv(topics_path)\
            .assign(Volume=lambda df: df['Volume'].apply(lambda x: int(x)))\
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


def text_to_wv_classname(text):
    return ''.join(e for e in text if e.isalnum() and not e.isdigit()).capitalize()


def serper_search(query, n):
    search = GoogleSerperAPIWrapper()
    search_results = search.results(query)
    search_results = [res for res in search_results['organic'] if 'youtube.com' not in res['link']]
    search_results = {'q': query, 'links': [res['link'] for res in search_results][:n]}
    urls = search_results['links']

    return urls


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True)
def get_content_near_vector(class_name: str, vector: list[float], n=1) -> dict:
    near_vector = {
        "vector": vector
    }

    result = wv_client.query\
        .get(f"{class_name}Content", ['url'])\
        .with_additional(['distance', 'id'])\
        .with_near_vector(near_vector)\
        .with_limit(n)\
        .do()
    
    return result


def scraper_scrape(url):
    #payload = {'api_key': SCRAPER_API_KEY, 'url': url, 'autoparse': True, 'render': True}
    #res = requests.get('http://api.scraperapi.com', params=payload, timeout=60)

    payload = {'api_key': SCRAPER_API_KEY, 'url': url, 'block_ads': 'true'}
    res = requests.get('https://app.scrapingbee.com/api/v1/', params=payload, timeout=60)
    html = res.text

    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('title').text
    raw_elements = partition_html(text=html)
    text = "\n\n".join([str(el) for el in raw_elements])
    elements = convert_to_dict(raw_elements)
    

    output = {
        'html': html,
        'elements': elements,
        'text': text,
        'title': title
    }
    
    return output


def scrape_content(urls: list[str], n=2) -> list[str]:
    def scrape_and_chunk_pdf(url, n, return_raw=False):
        r = requests.get(url, timeout=30, verify=False)
        file_name = url.split('/')[-1].replace('.pdf', '')
        path_name = LAMBDA_DATA_DIR + f'/{file_name}.pdf'
        with open(path_name, 'wb') as pdf:
            pdf.write(r.content)

        pages, meta = handle_pdf(pathlib.Path(path_name), n, url=url, return_raw=return_raw)
        os.remove(path_name)

        return pages, meta

    print(f"scrape_content n={n}")
    topic_content = []
    for url in urls:
        try:
            try:
                print(f"Trying to scrape as pdf: {url}")
                pages, meta = scrape_and_chunk_pdf(url, 100, return_raw=True)
                topic_content.append('\n\n'.join(pages)) 
            except Exception as e:
                print(e)
                output = scraper_scrape(url)
                topic_content.append(output['text'])
        except Exception as e:
            print(e)
            print(f"Skipping {url}")
            continue

        if len(topic_content) == n:
            break

    return topic_content


def mean_word_embedding(word_embeddings):
    """
    Calculate the mean word embedding from a list of word embeddings.

    :param word_embeddings: List of word embeddings, where each embedding is a list of floats.
    :return: Mean word embedding as a list of floats.
    """
    if not word_embeddings:
        raise ValueError("The list of word embeddings is empty.")

    # Get the length of embeddings
    embedding_length = len(word_embeddings[0])
    
    # Initialize a list with zeros for mean calculation
    mean_embedding = [0.0] * embedding_length
    
    # Sum all the embeddings
    for embedding in word_embeddings:
        if len(embedding) != embedding_length:
            raise ValueError("All word embeddings must be of the same length.")
        for i in range(embedding_length):
            mean_embedding[i] += embedding[i]
    
    # Divide by the number of embeddings to get the mean
    num_embeddings = len(word_embeddings)
    mean_embedding = [x / num_embeddings for x in mean_embedding]
    
    return mean_embedding



def topic_ecs(topic: str, ec_lib_name: str, user_email: str, customer_domain=None, top_n_ser=2, serper_api=True) -> dict:
    print(f'Getting Top {top_n_ser} Search Results')

    urls = []
    already_ranks = False
    n_search=10
    
    print('Using Serper API')
    urls = serper_search(topic, n_search)

    # Check if customer ranks for this topic and if so, ignore
    if customer_domain:
        if [d for d in urls if customer_domain in d]:
            print('Customer ranks for this topic')
            already_ranks = True

    if not urls:
        print('Issue with First Search. Trying Serper API Again')
        urls = serper_search(topic, n_search)
        
        if not urls:
            return {
                'topic': topic,
                'url': 'NONE',
                'distance': 1000,
                'score': 0,
                'already_ranks': already_ranks,
                'search_urls': ','.join(urls)
            }

    try:
        topic_content = scrape_content(urls, n=top_n_ser)

        if not topic_content:
            time.sleep(5)
            topic_content = scrape_content(urls, n=top_n_ser)
    except Exception as e:
        print(e)
        return {
            'topic': topic,
            'url': ','.join(urls),
            'distance': 1000,
            'score': 0,
            'already_ranks': already_ranks,
            'search_urls': ','.join(urls)
        }

    print('Getting Embeddings for Topic Results')
    text_embeddings = embedder.embed_documents(topic_content)
    topic_vector = mean_word_embedding(text_embeddings)


    print(f'Getting Most Similar Existing Content from {ec_lib_name}')
    ec_class_name, _ = get_wv_class_name(user_email, ec_lib_name)
    result = get_content_near_vector(ec_class_name, topic_vector)


    url = result['data']['Get'][f"{ec_class_name}Content"][0]['url']
    print(f'URL: {url}')
    distance = result['data']['Get'][f"{ec_class_name}Content"][0]['_additional']['distance']
    print(distance)
    score = 1 - distance
    print(f'Score: {score}')

    return {
        'topic': topic,
        'url': url,
        'distance': distance,
        'score': score,
        'already_ranks': already_ranks,
        'search_urls': ','.join(urls)
    }

  

def ecs(event, context):
    print(event)
    try:
        body = json.loads(event['body'])
    except:
        body = event['body']

    if 'sqs' in body:
       sqs = body['sqs']
    else:
       sqs = None

    topic = body['topic']
    ec_lib_name = body['ec_lib_name']
    user_email = body['user_email']
    customer_domain = body['customer_domain']
    top_n_ser = body['top_n_ser']
    serper_api = body['serper_api']

    # ECS
    ecs_result = topic_ecs(
        topic=topic,
        ec_lib_name=ec_lib_name,
        user_email=user_email,
        customer_domain=customer_domain,
        top_n_ser=top_n_ser,
        serper_api=serper_api
    )

    if sqs:
        queue = SQS(sqs)
        queue.send(ecs_result)
    else:
        return success(ecs_result)
    

def sample_ecs(event):
    print(event)
    try:
        body = json.loads(event['body'])
    except:
        body = event['body']

    email = body['user_email']
    top_n_ser = body['top_n_ser']
    ecs_concurrency = body['ecs_concurrency']
    ecs_job_id = body['ecs_job_id']

    # Get ECS Job
    ecs_job_res = get_bubble_object('ecs-job', ecs_job_id)
    ecs_job_json = ecs_job_res.json()['response']

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
    for content_url in ecs_job_json['content_urls'].split('\n'):
        out_body = {
            'email': email,
            'name': ec_lib_name,
            'doc_url': content_url
        }

        _ = lambda_client.invoke(
            FunctionName=f'foxscript-api-{STAGE}-upload_to_s3_cloud',
            InvocationType='RequestResponse',
            Payload=json.dumps({"body": out_body})
        )

    # Get Topics
    topics_df = pd.read_csv(local_batch_path)
    print(f"Topics Shape: {topics_df.shape}")
    topics = [t.split(' - ')[0] for t in topics_df.Keyword]
    topics = [t for t in topics if t][:10]
    
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
    domain_name = domain.split('.')[0]
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
        ecs_df = ecs_full_df.query('score >= 0.45')
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
    job_body = {
        'has_clustering_begun': True
    }
    res = update_bubble_object('ecs-job', ecs_job_id, job_body)

    cluster = cluster_keywords()
    try:
        input = {'input': local_ecs_path}
        cluster_path = cluster(input, keyword_col='topic', to_bubble=False)
    except:
        ecs_file_name = f'{domain_name}_ecs_full.csv'
        local_ecs_path = f'{LAMBDA_DATA_DIR}/{ecs_file_name}'
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
        'ecs_cluster_result': final_doc_id,
        'cost': keyword_doc_res.json()['response']['cost']
    }
    res = update_bubble_object('ecs-job', ecs_job_id, job_body)
    ecs_cluster_doc_id = res.json()['id']

    # Delete Wv Library
    delete_library(ec_class_name)
