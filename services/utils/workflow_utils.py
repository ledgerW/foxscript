import sys
sys.path.append('..')

import os
try:
  from dotenv import load_dotenv
  load_dotenv()
except:
  pass

import requests
from datetime import datetime
import time
import pandas as pd
import numpy as np
from utils.Kmeans import KMeans
from utils.cloud_funcs import cloud_scrape
from utils.general import SQS

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

try:
  from langchain.utilities import WikipediaAPIWrapper
except:
  pass

from langchain.utilities import GoogleSerperAPIWrapper

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

STAGE = os.getenv('STAGE')
WP_API_KEY = os.getenv('WP_API_KEY')

  

def get_top_n_search(query, n):
  search = GoogleSerperAPIWrapper()
  search_result = search.results(query)
  results = [res for res in search_result['organic'] if 'youtube.com' not in res['link']]
  
  return results[:n]


def get_ephemeral_vecdb(chunks, metadata):
  embeddings = OpenAIEmbeddings()
  
  return FAISS.from_texts(chunks, embeddings, metadatas=[metadata for _ in range(len(chunks))])


def get_context(query, llm, retriever, library=False):
  if library:
    prompt_template = """Use the following pieces of context to answer the question at the end.
  If the context doesn't directly answer the question, that's OK!  Even additional related information
  would be helpful! Include the sources you reference from the context in your answer.

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


# Cluster Sub-Topics Helpers
def get_content_embeddings(urls):
    embedder = OpenAIEmbeddings()
    topic_df = pd.DataFrame()

    # Distributed Scraping
    sqs = 'scrape{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
    queue = SQS(sqs)
    for url in urls:
        print(f'Scraping {url}')
        cloud_scrape(url, sqs=sqs)
        time.sleep(3)

    results = queue.collect(len(urls), max_wait=600)
    all_chunks = [result['chunks'].split('<SPLIT>') for result in results]
    print('Total chunks: {}'.format(len(all_chunks)))
    all_urls = [result['url'] for result in results]

    for url, chunks in zip(all_urls, all_chunks):
        print(f"Getting embeddings for {url}")
        text_embeddings = embedder.embed_documents(chunks)

        new_text_df = pd.DataFrame({
            'chunk': chunks,
            'url': url,
            'embedding': text_embeddings
        })

        topic_df = pd.concat([topic_df, new_text_df])

    topic_df = topic_df.drop_duplicates(subset=['chunk'])
    print('topic_df shape: {}'.format(topic_df.shape))

    return topic_df


def cluster(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    
    return kmeans


def get_topic_clusters(topic, top_n=10):
    search_results = get_top_n_search(topic, top_n)
    urls = [res['link'] for res in search_results]

    topic_df = get_content_embeddings(urls)

    embedding_matrix = np.vstack(topic_df.embedding.values)

    n_clusters = max(2, int(topic_df.shape[0]/10))
    print(f"Making {n_clusters} clusters")
    kmeans = cluster(embedding_matrix, n_clusters)
    topic_df['cluster'] = kmeans.labels_

    return topic_df


def get_cluster_results(topic_df, llm):
    print('Getting subtopic themes')
    input_word_cnt = 0
    output_word_cnt = 0
    samples = 5

    all_subtopics = ""
    clusters = topic_df.groupby('cluster', as_index=False).count().sort_values(by='chunk', ascending=False).cluster.values
    for i in clusters:
        this_cluster_df = topic_df[topic_df.cluster == i]
        n_samples = this_cluster_df.shape[0]

        sentences = "\n".join(this_cluster_df.sample(min(samples, n_samples)).chunk)
        prompt = f'Sentences:\n"""\n{sentences}\n"""\n\nWhat do the sentences above have in common? Give them a descriptive theme name.\n\nTheme:'

        # TODO: FoxLLM fallbacks
        res = llm.invoke(prompt)
        theme = res.content.replace("\n", "")

        subtopic = f"Sub-topic {i}: {theme}\n"
        subtopic = subtopic + f"N Samples: {n_samples}\n"

        return_n_samples = min(3, n_samples)
        sample_cluster_rows = topic_df[topic_df.cluster == i].sample(return_n_samples)
        for j in range(return_n_samples):
            subtopic = subtopic + sample_cluster_rows.chunk.values[j] + '\n'

        subtopic = subtopic + ("-" * 10) + '\n'

        all_subtopics = all_subtopics + subtopic
        input_word_cnt = input_word_cnt + len(prompt.replace('\n', ' ').split(' '))
        output_word_cnt = output_word_cnt + len(subtopic.replace('\n', ' ').split(' '))

    return all_subtopics, input_word_cnt, output_word_cnt


def get_search_snippets(query):
    search = GoogleSerperAPIWrapper()
    search_result = search.results(query)
    
    snippets = []
    sources = []
    if 'knowledgeGraph' in search_result:
        snippets.append(search_result['knowledgeGraph']['description'])
        sources.append(search_result['knowledgeGraph']['descriptionLink'])
    if 'answerBox' in search_result:
        snippets.append(search_result['answerBox']['snippet'])
        sources.append(search_result['answerBox']['link'])
    if 'organic' in search_result:
        for organic in search_result['organic'][:3]:
            snippets.append(organic['snippet'])
            sources.append(organic['link'])
    if 'peopleAlsoAsk' in search_result:
        for also_ask in search_result['peopleAlsoAsk'][:5]:
            snippets.append(also_ask['snippet'])
            sources.append(also_ask['link'])

    return snippets, sources


def get_wiki_snippets(query):
  wiki = WikipediaAPIWrapper(top_k_results=3)
  wiki_pages = wiki.wiki_client.search(query)

  snippets = []
  sources = []
  for page in wiki_pages:
    sum = wiki.wiki_client.summary(page)
    
    if sum != '':
      sum = ' '.join(sum.split(' ')[:75])
      snippets.append(sum)
      sources.append(f'Wikipedia - {page}')


    if len(snippets) == 3:
      break

  return snippets, sources



# WORDPRESS STUFF
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
      img_url = 'FUNC TO GET AN IMG URL'
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