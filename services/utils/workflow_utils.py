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
import json
import pandas as pd
import numpy as np
from utils.Kmeans import KMeans
from utils.cloud_funcs import cloud_scrape, cloud_google_search
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

STAGE = os.getenv('STAGE')
WP_API_KEY = os.getenv('WP_API_KEY')

  

def get_top_n_search(query, n, sqs=None):
    try:
        res = cloud_google_search(q=query, n=None, sqs=sqs)
        res_body = json.loads(res['Payload'].read().decode("utf-8"))
        results = json.loads(res_body['body'])['results']
        results = [res for res in results if 'youtube.com' not in res['link']]
    except:
        print('Issue with Cloud Google Search. Using Serper API')
        search = GoogleSerperAPIWrapper()
        results = search.results(query)
        results = [res for res in results['organic'] if 'youtube.com' not in res['link']]

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
        time.sleep(1)

    results = queue.collect(len(urls), max_wait=420)
    all_chunks = [result['chunks'].split('<SPLIT>') for result in results]
    print('Total chunks: {}'.format(sum([len(l) for l in all_chunks])))
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
    print("Length of urls: {}".format(len(urls)))

    topic_df = get_content_embeddings(urls)

    embedding_matrix = np.vstack(topic_df.embedding.values)

    n_clusters = max(2, int(topic_df.shape[0]/10))
    n_clusters = min(20, n_clusters)
    print(f"Making {n_clusters} clusters")
    kmeans = cluster(embedding_matrix, n_clusters)
    topic_df['cluster'] = kmeans.labels_

    return topic_df


def get_cluster_results(topic_df, LLM):
    print('Getting subtopic themes')
    input_word_cnt = 0
    output_word_cnt = 0

    all_subtopics = ""
    clusters = topic_df.groupby('cluster', as_index=False).count().sort_values(by='chunk', ascending=False).cluster.values
    existing_themes = ''
    for idx, i in enumerate(clusters):
        this_cluster_df = topic_df[topic_df.cluster == i]
        n_samples = this_cluster_df.shape[0]

        if n_samples > 0:
            sentences = ''
            for url in this_cluster_df.url.unique():
                sentences = sentences + f"\nSource: {url}\n"
                this_url_df = this_cluster_df.query('url == @url').reset_index(drop=True)
                for j in range(this_url_df.shape[0]):
                    sentences = sentences + '- ' + this_url_df.chunk.values[j] + '\n'
            
            prompt = """Sentences:
    {}

    Existing Themes:
    {}
    
    You have two jobs.
    1) Come up with a short, descriptive, specific theme name for the sentences provided above. But you can't reuse an Existing Theme name.
    2) Write a thorough, detail-oriented distilation of the sentences. Be sure to capture any specific numbers or figures.
    The details are important! Think of this task as distilling all the information without leaving out any important details.
    Prefer thoughness over brevity here.

    Follow the template below for your output.
    
    Theme: [theme name]
    Key Elements:
    [distilation of sentences]
    
    Sources:
    [list of source urls]""".format(sentences, existing_themes)

            # FoxLLM fallbacks, if necessary
            res = None
            try:
                res = LLM.llm.invoke(prompt)
            except:
                for llm in LLM.fallbacks:
                    print(f'fallback to {llm}')
                    LLM.llm = llm
                    try:
                        res = LLM.llm.invoke(prompt)
                        if res:
                            break
                    except:
                        continue
            
            theme_and_summary = res.content
            theme = theme_and_summary.split('Theme:')[1].split('Key Elements:')[0].replace('\n', '').strip()
            existing_themes = existing_themes + f"\n{theme}"

            subtopic = f"Subtopic {idx+1}\n"
            subtopic = subtopic + f"Sections of text found: {n_samples}\n"
            subtopic = subtopic + f"{theme_and_summary}\n"

            subtopic = subtopic + '\n' + ("_" * 3) + '\n\n'

            all_subtopics = all_subtopics + subtopic
            input_word_cnt = input_word_cnt + len(prompt.replace('\n', ' ').split(' '))
            output_word_cnt = output_word_cnt + len(subtopic.replace('\n', ' ').split(' '))

    return all_subtopics, input_word_cnt, output_word_cnt


def get_cluster_results_by_source(topic_df, LLM):
    print('Getting subtopic themes')
    input_word_cnt = 0
    output_word_cnt = 0

    all_subtopics = ""
    clusters = topic_df.groupby('cluster', as_index=False).count().sort_values(by='chunk', ascending=False).cluster.values
    existing_themes = ''
    for idx, i in enumerate(clusters):
        print(f"Cluster: {idx}")
        this_cluster_df = topic_df[topic_df.cluster == i]
        n_samples = this_cluster_df.shape[0]

        if n_samples > 0:
            theme_url_statements = ''
            for url in this_cluster_df.url.unique():
                sentences = f"\nSource: {url}\n"
                this_url_df = this_cluster_df.query('url == @url').reset_index(drop=True)
                for j in range(this_url_df.shape[0]):
                    sentences = sentences + '- ' + this_url_df.chunk.values[j] + '\n'

                # Prompt for Subtopic URL Source
                url_prompt = """Sentences:
    {}
    
    Your job is to write a thorough, detail-oriented distilation of the sentences above. Be sure to capture any specific numbers or figures.
    The details are important! Think of this task as distilling all the information without leaving out any important details.
    Prefer thoughness over brevity here.

    Follow the example below for your output...
    
    Example Output:
    Source: [url source from above]
    Key Elements:
    - detailed distillation of
    - of sentences above in bullet format
    

    Output:""".format(sentences)
                
                # FoxLLM fallbacks, if necessary
                res = None
                try:
                    res = LLM.llm.invoke(url_prompt)
                except:
                    for llm in LLM.fallbacks:
                        print(f'fallback to {llm}')
                        LLM.llm = llm
                        try:
                            res = LLM.llm.invoke(url_prompt)
                            if res:
                                break
                        except:
                            continue

                theme_url_statements = theme_url_statements + res.content + '\n\n'
                    
            
            # Prompt for Cluster/Subtopic Theme
            theme_prompt = """Statements:
    {}

    Existing Themes:
    {}
    
    You have two jobs.
    1) Come up with a short, descriptive, specific theme name for the statements provided above. But you can't reuse an Existing Theme name.
    You should ignore the Source URLs. Return only the Theme Name.
    
    Theme Name:""".format(theme_url_statements, existing_themes)

            # FoxLLM fallbacks, if necessary
            res = None
            try:
                res = LLM.llm.invoke(theme_prompt)
            except:
                for llm in LLM.fallbacks:
                    print(f'fallback to {llm}')
                    LLM.llm = llm
                    try:
                        res = LLM.llm.invoke(theme_prompt)
                        if res:
                            break
                    except:
                        continue
            
            theme_name = res.content
            existing_themes = existing_themes + f"\n{theme_name}"

            subtopic = f"Subtopic {idx+1}\n"
            subtopic = subtopic + f"Sections of text found: {n_samples}\n"
            subtopic = subtopic + f"{theme_name}\n"
            subtopic = subtopic + f"{theme_url_statements}"

            subtopic = subtopic + '\n' + ("_" * 3) + '\n'

            all_subtopics = all_subtopics + subtopic
            input_word_cnt = input_word_cnt + len(url_prompt.replace('\n', ' ').split(' '))
            input_word_cnt = input_word_cnt + len(theme_prompt.replace('\n', ' ').split(' '))
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
#def get_yt_url(query):
#  vid_search = VideosSearch(query, limit = 3)
#
#  return vid_search.result()['result'][0]['link']


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