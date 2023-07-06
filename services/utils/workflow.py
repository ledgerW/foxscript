import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()


STAGE = os.getenv('STAGE')
WP_API_KEY = os.getenv('WP_API_KEY')


import json
import time
import requests
import pathlib
import boto3
from bs4 import BeautifulSoup

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.experimental.plan_and_execute.schema import (
    Plan,
    PlanOutputParser,
    Step,
)

from utils.scrapers.base import Scraper
from utils.content import handle_pdf

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

lambda_client = boto3.client('lambda')


class ArticleParser(PlanOutputParser):
  def parse(self, text: str) -> Plan:
    steps = [Step(value=v) for v in text.split('ARTICLE:\n')[1:]]
    return Plan(steps=steps)


class GeneralScraper(Scraper):
  blog_url = None
  source = None
  base_url = None

  def __init__(self):
    self.driver = self.get_selenium()


  def scrape_post(self, url=None):
    self.driver.get(url)
    time.sleep(5)
    html = self.driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
    self.driver.quit()

    soup = BeautifulSoup(html, 'lxml')

    return soup
  

def get_top_n_search(query, n):
  search = GoogleSerperAPIWrapper()
  search_result = search.results(query)
  
  return search_result['organic'][:n]


def text_splitter(text, n, tokenizer):
  chunks = []
  chunk = ''
  sentences = [s.strip().replace('\n', ' ') for s in text.split('.')]
  for s in sentences:
    # start new chunk
    if chunk == '':
      chunk = s
    else:
      chunk = chunk + ' ' + s
    
    chunk_len = len(tokenizer.encode(chunk))
    if chunk_len >= 0.9*n:
      chunks.append(chunk)
      chunk = ''

  if chunk != '':
    chunks.append(chunk)
  
  return chunks


def scrape_and_chunk_pdf(url, n, tokenizer):
  r = requests.get(url)
  with open('tmp.pdf', 'wb') as pdf:
    pdf.write(r.content)

  return handle_pdf(pathlib.Path('tmp.pdf'), n, tokenizer)


def scrape_and_chunk(url, token_size, tokenizer):
  if url.endswith('.pdf'):
    chunks, pages, meta = scrape_and_chunk_pdf(url, 100, tokenizer)
    
    return chunks
  else:
    scraper = GeneralScraper()
    soup = scraper.scrape_post(url)

    for script in soup(["script", "style"]):
      script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    results = "\n".join(chunk for chunk in chunks if chunk)

    return text_splitter(results, token_size, tokenizer)
  

def cloud_scrape(url, sqs=None, query=None):
  _ = lambda_client.invoke(
    FunctionName=f'foxscript-data-{STAGE}-scraper',
    InvocationType='Event',
    Payload=json.dumps({"body": {
        'url': url,
        'sqs': sqs,
        'query': query
      }})
  )


def cloud_research(url, sqs=None, query=None):
  _ = lambda_client.invoke(
    FunctionName=f'foxscript-data-{STAGE}-researcher',
    InvocationType='Event',
    Payload=json.dumps({"body": {
        'url': url,
        'sqs': sqs,
        'query': query
      }})
  )


def get_ephemeral_vecdb(chunks, metadata):
  embeddings = OpenAIEmbeddings()
  
  return FAISS.from_texts(chunks, embeddings, metadatas=[metadata for _ in range(len(chunks))])


def get_sources_context(query, llm, retriever):
  vec_qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  res = vec_qa({'question': query})
  
  return '\n'.join(['source: {}'.format(res['sources']), res['answer']])


def get_context(query, llm, retriever):
  vec_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  res = vec_qa({'query': query})
  
  return res['result']


def get_library_retriever():
  embeddings = OpenAIEmbeddings()

  # Load from Local
  db = FAISS.load_local("vecstore_backup", embeddings)
  
  return db.as_retriever(search_kwargs={"k": 4})


def get_article_img(article):
  key_fig_1 = article.split('### Who')[1].split('\n')[1]

  search = GoogleSerperAPIWrapper(type="images")
  results = search.results(key_fig_1)

  img_url = results['images'][0]['imageUrl']
  for img in results['images']:
    if (img['imageWidth'] > img['imageHeight']) and ('.jpg' in img['imageUrl']):
        img_url = img['imageUrl']
        break
    
  return img_url


def upload_wp_media_image(img_url):
    url = 'https://public-api.wordpress.com/rest/v1.1/sites/chatterboxoffice.com/media/new'
    headers = {
        'Authorization': f'Bearer {WP_API_KEY}',
    }
    data = {
        'media_urls': img_url
    }

    response = requests.post(url, headers=headers, data=data)

    return response.json()['media'][0]['ID']


def create_wp_post(article):
    try:
      img_url = get_article_img(article)
      img_id = upload_wp_media_image(img_url)
    except:
      img_url = "https://chatterboxoffice.com/wp-content/uploads/2023/06/cropped-transparent-logo.png"
      img_id = upload_wp_media_image(img_url)

    post_url = 'https://public-api.wordpress.com/rest/v1.1/sites/chatterboxoffice.com/posts/new'
    post_headers = {
        'Authorization': f'Bearer {WP_API_KEY}',
    }
    post_data = {
        'title': article.split('\n')[0].replace('#', '').strip(),
        'content': "\n".join(article.split('\n')[1:]).strip(),
        'featured_image': img_id,
        'status': 'draft',
        'author': 'chatterboxofficestaff',
        'format': 'standard'
    }

    post_response = requests.post(post_url, headers=post_headers, data=post_data)

    return post_response.json()