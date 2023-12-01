import sys
sys.path.append('..')

import os
import json
import requests
import pathlib
from unstructured.partition.html import partition_html

from newspaper import Article

from utils.FoxLLM import FoxLLM, az_openai_kwargs, openai_kwargs
from utils.content import text_splitter
from utils.response_lib import *
from utils.scrapers.base import Scraper
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

from utils.content import handle_pdf

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")


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

    def __init__(self):
        self.driver = self.get_selenium()
        self.driver.set_page_load_timeout(180)


    def scrape_post(self, url=None):
        self.driver.get(url)

        page_content = self.driver.page_source
        elements = partition_html(text=page_content)
        text = "\n\n".join([str(el) for el in elements])
        self.driver.quit()

        return text


def scrape_and_chunk_pdf(url, n, tokenizer):
  r = requests.get(url, timeout=10)
  with open(LAMBDA_DATA_DIR + '/tmp.pdf', 'wb') as pdf:
    pdf.write(r.content)

  return handle_pdf(pathlib.Path(LAMBDA_DATA_DIR, 'tmp.pdf'), n, tokenizer)


def scrape_and_chunk(url, token_size, tokenizer, sentences=False):
  try:
    chunks, pages, meta = scrape_and_chunk_pdf(url, 100, tokenizer)
    
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
            text = scraper.scrape_post(url)
    else:
        print('processing non-news source')
        scraper = GeneralScraper()
        text = scraper.scrape_post(url)
    
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    results = "\n".join(chunk for chunk in chunks if chunk)

    return text_splitter(results, token_size, tokenizer, sentences)
  

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

    # Scrape
    chunks = scrape_and_chunk(url, 100, tokenizer)
    
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
      LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-35-16k', temp=1.0)

      chunks = scrape_and_chunk(url, 100, tokenizer)
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



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--url', default=None, type=str)
  parser.add_argument('--query', default=None, type=str)
  args, _ = parser.parse_known_args()

  print(args)

  #llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1.0)
  chunks = scrape_and_chunk(args.url, 100, tokenizer)
  vec_db = get_ephemeral_vecdb(chunks, {'source': args.url})
  research_results = get_context(args.query, llm, vec_db.as_retriever())

  print(research_results)
