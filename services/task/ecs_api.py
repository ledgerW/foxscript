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
import requests
from unstructured.partition.html import partition_html
from unstructured.staging.base import convert_to_dict
from bs4 import BeautifulSoup
from utils.general import SQS
from utils.response_lib import *
from utils.weaviate_utils import wv_client, get_wv_class_name
from utils.content import handle_pdf


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
   LAMBDA_DATA_DIR = '.'
else:
   LAMBDA_DATA_DIR = '/tmp'

embedder = OpenAIEmbeddings(model="text-embedding-3-large")


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
