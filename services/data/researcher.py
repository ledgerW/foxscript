import sys
sys.path.append('..')

import os
import json
import time
import requests
import pathlib
from bs4 import BeautifulSoup

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from utils.response_lib import *
from utils.scrapers.base import Scraper
from utils.general import SQS
from utils.content import handle_pdf

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
  LAMBDA_DATA_DIR = '.'
else:
  LAMBDA_DATA_DIR = '/tmp'


class GeneralScraper(Scraper):
    blog_url = None
    source = None
    base_url = None

    def __init__(self):
        self.driver = self.get_selenium()
        self.driver.set_page_load_timeout(10)


    def scrape_post(self, url=None):
        self.driver.get(url)
        time.sleep(5)
        html = self.driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        self.driver.quit()

        soup = BeautifulSoup(html, 'lxml')

        return soup


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
  r = requests.get(url, timeout=10)
  with open(LAMBDA_DATA_DIR + '/tmp.pdf', 'wb') as pdf:
    pdf.write(r.content)

  return handle_pdf(pathlib.Path(LAMBDA_DATA_DIR, 'tmp.pdf'), n, tokenizer)


def scrape_and_chunk(url, token_size, tokenizer):
  try:
    chunks, pages, meta = scrape_and_chunk_pdf(url, 100, tokenizer)
    
    return chunks
  except:
    scraper = GeneralScraper()
    soup = scraper.scrape_post(url)

    for script in soup(["script", "style"]):
      script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    results = "\n".join(chunk for chunk in chunks if chunk)

    return text_splitter(results, token_size, tokenizer)
  

def get_ephemeral_vecdb(chunks, metadata):
  embeddings = OpenAIEmbeddings()
  
  return FAISS.from_texts(chunks, embeddings, metadatas=[metadata for _ in range(len(chunks))])


def get_context(query, llm, retriever):
  prompt_template = """Use the following pieces of context to answer the question at the end.
  If the context doesn't directly answer the question, that's OK!  Even additional related information
  would be helpful! 

  {context}

  Question: {question}
  Helpful Answer:"""
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )

  chain_type_kwargs = {"prompt": PROMPT}
  vec_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
  res = vec_qa({'query': query})

  return res['result']


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

    if 'query' in body:
       query = body['query']
    else:
       query = None

    # Scrape
    chunks = scrape_and_chunk(url, 100, tokenizer)

    result = {'url': url, 'query': query, 'chunks': chunks}

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
    try:
      llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.8)
      chunks = scrape_and_chunk(url, 100, tokenizer)
      vec_db = get_ephemeral_vecdb(chunks, {'source': url})
      context = get_context(query, llm, vec_db.as_retriever())
    except:
      context = 'Problem analyzing source.'

    result = f"query: {query}\n" + f"source: {url}\n" + context + '\n'

    if sqs:
       queue = SQS(sqs)
       queue.send(result)
    else:
       return success(result)



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--url', default=None, type=str)
  args, _ = parser.parse_known_args()

  chunks = scrape_and_chunk(args.url, 100, tokenizer)

  print(len(chunks))
  print(chunks[:3])
