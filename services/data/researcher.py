import sys
sys.path.append('..')

import os
import json
import requests
import pathlib
from unstructured.partition.html import partition_html
from unstructured.staging.base import convert_to_dict
from bs4 import BeautifulSoup

from newspaper import Article

from utils.FoxLLM import FoxLLM, az_openai_kwargs, openai_kwargs
from utils.content import text_splitter, handle_pdf
from utils.response_lib import *
from scrapers.base import Scraper
from utils.workflow_utils import get_ephemeral_vecdb, get_context


try:
  from utils.general import SQS
except:
  pass

try:
  from dotenv import load_dotenv
  load_dotenv()
except:
  pass


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
  LAMBDA_DATA_DIR = '.'
else:
  LAMBDA_DATA_DIR = '/tmp'


def is_news_source(url):
    try:
       f = open("news_sources.txt", "r")
    except:
       f = open("data/news_sources.txt", "r")
    sources = f.read().split('\n')

    print('news source:')
    print([src for src in sources if src in url])
    
    return len([src for src in sources if src in url]) > 0


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

        return {'q': q, 'links': all_links}


def scrape_and_chunk_pdf(url, n, return_raw=False):
    r = requests.get(url, timeout=10)
    file_name = url.split('/')[-1].replace('.pdf', '')
    path_name = LAMBDA_DATA_DIR + f'/{file_name}.pdf'
    with open(path_name, 'wb') as pdf:
        pdf.write(r.content)

    return handle_pdf(pathlib.Path(path_name), n, url=url, return_raw=return_raw)


def scrape_and_chunk(url, token_size, sentences=False, chunk_overlap=10, return_raw=False):
    try:  
        if return_raw:
            pages, meta = scrape_and_chunk_pdf(url, 100, return_raw=return_raw)
            return '\n'.join([meta['url'], meta['title'], meta['author']]) + '\n\n'.join(pages) 
        else:
            chunks, pages, meta = scrape_and_chunk_pdf(url, 100, return_raw=return_raw)
            return chunks
    except:
        if is_news_source(url):
            print('processing news source')
            article = Article(url=url)
            try:
                article.download()
                article.parse()
                text = article.text
            except:
                print('issue with news source - processing as non-news source')
                scraper = GeneralScraper()
                output, _ = scraper.scrape_post(url)
                text = output['text']
        else:
            print('processing non-news source')
            scraper = GeneralScraper()
            output, _ = scraper.scrape_post(url)
            text = output['text']
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        results = "\n".join(chunk for chunk in chunks if chunk)

        return text_splitter(results, token_size, sentences, chunk_overlap=chunk_overlap)
  

def google_search(event, context):
    print(event)
    try:
        body = event['body']
    except:
        body = json.loads(event['body'])

    q = body['q']

    if 'n' in body:
       n = body['n']
    else:
       n = 50

    if 'sqs' in body:
       sqs = body['sqs']
    else:
       sqs = None

    # Scrape Google Search
    scraper = GeneralScraper(is_google_search=True)
    result = scraper.google_search(q)   # returns {q:str, links:[str]}
    result['links'] = result['links'][:n]

    if sqs:
       queue = SQS(sqs)
       queue.send(result)
    else:
       return success(result)


def scrape(event, context):
    print(event)
    try:
        body = event['body']
    except:
        body = json.loads(event['body'])

    if 'url' in body:
       url = body['url']

    if 'sqs' in body:
       sqs = body['sqs']
    else:
       sqs = None

    if 'chunk_overlap' in body:
        chunk_overlap = body['chunk_overlap']
    else:
        chunk_overlap = 10

    if 'return_raw' in body:
        return_raw = body['return_raw']
    else:
        return_raw = False

    # Scrape
    chunks = scrape_and_chunk(url, 100, chunk_overlap=chunk_overlap, return_raw=return_raw)
    
    if not return_raw:
        chunks = '<SPLIT>'.join(chunks)

    result = {'url': url, 'chunks': chunks}

    if sqs:
       queue = SQS(sqs)
       queue.send(result)
    else:
       return success(result)
    

def research(event, context):
    print(event)
    try:
        body = event['body']
    except:
        body = json.loads(event['body'])

    if 'url' in body:
       url = body['url']

    if 'sqs' in body:
       sqs = body['sqs']
    else:
       sqs = None

    if 'query' in body:
       query = body['query']
    else:
       query = None

    # Scrape and Research
    research_results = None
    try:
      LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-35-16k', temp=0.1)

      chunks = scrape_and_chunk(url, 100)
      vec_db = get_ephemeral_vecdb(chunks, {'source': url})

      try:
        research_results = get_context(query, LLM.llm, vec_db.as_retriever())
      except:
        for llm in LLM.fallbacks:
          try:
            research_results = get_context(query, llm, vec_db.as_retriever())
            if research_results:
              break
          except:
              continue

      if not research_results:
        research_results = "Problem analyzing source."

    except Exception as error:
      print("Problem analyzing source: ", error)
      research_results = "Problem analyzing source."

    result = f"query: {query}\n" + f"source: {url}\n" + research_results + '\n'

    if sqs:
        queue = SQS(sqs)
        queue.send({
            'output': result,
            'input_word_cnt': len(query.split(' ')),
            'output_word_cnt': len(result.split(' '))
        })
    else:
        return success({
            'output': result,
            'input_word_cnt': len(query.split(' ')),
            'output_word_cnt': len(result.split(' '))
        })
