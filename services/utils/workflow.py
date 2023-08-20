import sys
sys.path.append('..')

import os
try:
  from dotenv import load_dotenv
  load_dotenv()
except:
  pass

import json
import requests
import boto3

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

STAGE = os.getenv('STAGE')
WP_API_KEY = os.getenv('WP_API_KEY')


def get_wv_class_name(email, name):
    domain = email.split('@')[1].split('.')[0]
    username = email.split('@')[0].replace('.', '').capitalize()
    account = f"{username}{domain}"
    name = name.capitalize()

    cls_name = f"{account}{name}"

    return cls_name, account
  

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


def get_sources_context(query, llm, retriever):
  vec_qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  res = vec_qa({'question': query})
  
  return '\n'.join(['source: {}'.format(res['sources']), res['answer']])


def get_ephemeral_vecdb(chunks, metadata):
  embeddings = OpenAIEmbeddings()
  
  return FAISS.from_texts(chunks, embeddings, metadatas=[metadata for _ in range(len(chunks))])


def get_context(query, llm, retriever, library=False):
  if library:
    prompt_template = """Use the following pieces of context to answer the question at the end.
  If the context doesn't directly answer the question, that's OK!  Even additional related information
  would be helpful!  Include the sources you reference from the context in your answer.

  {context}

  Question: {question}
  
  Helpful response in the below format:
  Sources: [sources from context here]
  [response here]"""
  else:
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