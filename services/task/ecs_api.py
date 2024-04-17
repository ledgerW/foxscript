import sys
sys.path.append('..')

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import json
import boto3
import numpy as np
import pathlib
import requests
from unstructured.partition.html import partition_html
from unstructured.staging.base import convert_to_dict
from bs4 import BeautifulSoup
from utils.bubble import (
    create_bubble_object,
    get_bubble_object,
    update_bubble_object,
    get_bubble_doc,
    delete_bubble_object,
    upload_bubble_file
)
from utils.general import SQS
from utils.response_lib import *
from utils.weaviate_utils import wv_client, get_wv_class_name
from utils.content import handle_pdf
#from utils.workflow_utils import get_top_n_search
from scrapers.base import Scraper


from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
   LAMBDA_DATA_DIR = '.'
else:
   LAMBDA_DATA_DIR = '/tmp'

embedder = OpenAIEmbeddings(model="text-embedding-3-large")


class GeneralScraper(Scraper):
    blog_url = None
    source = None
    base_url = None

    def __init__(self, is_google_search: bool=False):
        self.is_google_search = is_google_search

        if not self.is_google_search:
            self.driver = self.get_selenium()
            self.driver.set_page_load_timeout(180)


    def scrape_post(self, url: str=None):
        self.driver.get(url)

        html = self.driver.page_source
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
        
        self.driver.quit()

        return output, raw_elements
    

    def google_search(self, q: str):
        from fake_useragent import UserAgent
        ua = UserAgent()
        header = {'User-Agent':str(ua.random)}

        url = f"https://www.google.com/search?q={q.replace(' ', '+')}"
        html = requests.get(url, headers=header)
        
        soup = BeautifulSoup(html.content, 'html.parser')

        all_links = []
        for a in soup.select("a:has(h3)"):
            try:
                if 'https://' in a['href'] or 'http://' in a['href']:
                    link = a['href'].split('#')[0]
                    if link not in all_links:
                        all_links.append(link)
            except:
                pass

        all_links = [url.replace('/url?q=','').split('&sa')[0] for url in all_links]
        return {'q': q, 'links': all_links}


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


def scrape_content(urls: list[str]) -> list[str]:
    def scrape_and_chunk_pdf(url, n, return_raw=False):
        r = requests.get(url, timeout=10)
        file_name = url.split('/')[-1].replace('.pdf', '')
        path_name = LAMBDA_DATA_DIR + f'/{file_name}.pdf'
        with open(path_name, 'wb') as pdf:
            pdf.write(r.content)

        return handle_pdf(pathlib.Path(path_name), n, url=url, return_raw=return_raw)

    topic_content = []
    for url in urls:
        try:
            pages, meta = scrape_and_chunk_pdf(url, 100, return_raw=True)
            topic_content.append('\n\n'.join(pages)) 
        except:
            scraper = GeneralScraper()
            print(f'Scraping {url}')
            output, _ = scraper.scrape_post(url)
            topic_content.append(output['text'])

    return topic_content


def topic_ecs(topic: str, ec_lib_name: str, user_email: str, customer_domain=None, top_n_ser=2) -> dict:
    print(f'Getting Top {top_n_ser} Search Results')
    already_ranks = False
    urls = []
    n=10
    attempt = 0
    while not urls and attempt < 3:
        #search_results = get_top_n_search(topic, n=10)
        try:
            scraper = GeneralScraper(is_google_search=True)
            search_results = scraper.google_search(topic)   # returns {q:str, links:[str]}
            search_results['links'] = search_results['links'][:n]
            urls = search_results['links']
        except:
            print('Issue with Cloud Google Search. Using Serper API')
            search = GoogleSerperAPIWrapper()
            search_results = search.results(topic)
            search_results = [res for res in search_results['organic'] if 'youtube.com' not in res['link']]
            result = {'q': topic, 'links': [res['link'] for res in search_results][:n]}
            urls = search_results['links']

        # Check if customer ranks for this topic and if so, ignore
        if customer_domain:
            if [d for d in urls if customer_domain in d]:
                print('Customer ranks for this topic')
                already_ranks = True

        attempt += 1

    if not urls:
        return None
    else:
        urls = urls[:2]

    try:
        topic_content = scrape_content(urls)
    except:
        return None

    print('Getting Embeddings for Topic Results')
    text_embeddings = embedder.embed_documents(topic_content)
    topic_vector = np.average(text_embeddings, axis=0, keepdims=True).tolist()[0]

    print(f'Getting Most Similar Existing Content from {ec_lib_name}')
    ec_class_name, _ = get_wv_class_name(user_email, ec_lib_name)
    result = get_content_near_vector(ec_class_name, topic_vector)

    url = result['data']['Get'][f"{ec_class_name}Content"][0]['url']
    print(f'URL: {url}')
    distance = result['data']['Get'][f"{ec_class_name}Content"][0]['_additional']['distance']
    print(distance)
    score = 1 - distance
    print(f'Score: {score}')

    return {'topic': topic, 'url': url, 'distance': distance, 'score': score, 'already_ranks': already_ranks}

  

    

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

    # ECS
    ecs_result = topic_ecs(
        topic=topic,
        ec_lib_name=ec_lib_name,
        user_email=user_email,
        customer_domain=customer_domain,
        top_n_ser=top_n_ser
    )

    if sqs:
        queue = SQS(sqs)
        queue.send(ecs_result)
    else:
        return success(ecs_result)
