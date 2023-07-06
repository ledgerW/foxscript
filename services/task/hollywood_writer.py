import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()


STAGE = os.getenv('STAGE')
WP_API_KEY = os.getenv('WP_API_KEY')
BUCKET = os.getenv('BUCKET')


import json
import argparse
import time
import requests
import pathlib
import boto3
from datetime import datetime
from bs4 import BeautifulSoup
from youtubesearchpython import VideosSearch

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.experimental.plan_and_execute.schema import (
    Plan,
    PlanOutputParser,
    Step,
)

from utils.general import SQS
from utils.scrapers.base import Scraper
from utils.scrapers import EWScraper
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


def get_breakdown_chain(llm):
    breakdown_template = """You are a Senior Staff Writer for ChatterBoxOffice, a daily hollywood and 
  entertainment news website that publishes articles backed by research.

  ChatterBoxOffice starts with the subject of an existing ARTICLE, then does more research on the subject, 
  and then writes a fun article that is better than the original ARTICLE.

  Our research focuses on:
  1. discovering the broader context of the ARTICLE by learning what led to this, what else is happening now,
  and where might this lead in the future.
  2. how is what's happening in the ARTICLE being reporting online and on social media
  3. any relevant busines or finanical implications related to what is discussed in the ARTICLE
  4. what else are the key figures in the ARTICLE doing, currently or in the future
  5. opinions of subject matter experts on what's happening in the ARTICLE

  Given the ARTICLE below...
  ARTICLE:
  {input}

  Perform the following tasks in the following format:
  Summary:
  Summarize the ARTICLE

  Key Figures:
  - bullet point list of
  - the top few key people or companies or organizations

  Key Details:
  - bullet point list of
  - key details and
  - key numbers
  
  Key Quotes:
  - bullet point list of
  - key quotes

  Movie or TV Series:
  If the ARTICLE refers to a movie or tv series, then write the name and season number of the movie or tv series here.

  Research Questions:
  1. ordered list of
  2. 8-12 questions we want to research
  
  The questions will be used as internet search queries.
  Use specific names from the ARTICLE in your questions to get better results.
    """

    input_vars = ['input']

    breakdown_prompt = PromptTemplate(
        input_variables=input_vars, template=breakdown_template
    )

    return LLMChain(llm=llm, prompt=breakdown_prompt, verbose=True)


def parse_queries(queries):
    questions = queries.split('Research Questions:')[1]
    questions = [f for f in questions.split('\n') if f != '']

    return questions


def get_library_retriever():
  embeddings = OpenAIEmbeddings()

  # Load from Local
  db = FAISS.load_local("vecstore_backup", embeddings)
  
  return db.as_retriever(search_kwargs={"k": 4})


def get_article_chain(llm):
    _article_draft_prompt = """You are writing an article for a daily Hollywood and 
  entertainment news website that publishes articles backed by research.

  You are a 30 year old female New Yorker.
  You are snarky and witty and while you love Hollywood, you've been doing this job for a while, so you're a little cynical, but still funny.
  You usually have polite criticism.
  DO NOT refer to yourself in your writing.
  DO NOT start sentences with "As a writer..." or "as a snarky New Yorker...", etc... or use that pattern at all!

  Your job is to use the BREAKDOWN and RESEARCH below to write a well researched article.

  If the RESEARCH seems unrelated to what's in the BREAKDOWN, you can ignore it.

  BREAKDOWN:
  {breakdown}

  RESEARCH:
  {research}

  You get bonus points if you can accomplish the following in your article:
  - Draw an interesting but non-obvious connection between something in the RESEARCH and the BREAKDOWN
  - In the Scoop section, use markdown links to cite your RESEARCH sources, like this: [this is a statement in the Scoop](source url from RESEARCH)

  Write the article using markdown according to the following pattern:

  # Clever Title With Emoji Here

  ### Who
  - Bullet Point List
  - of top 5 or fewer Key Figures in the article

  ### TLDR
  - Bullet Point List
  - of Key Points of the article

  ### Quotes
  Use markdown quotation blocks for important quotes in the article

  ### Video
  [YT_PLACEHOLDER - DON'T REMOVE]

  ### Scoop
  - This is the main body of the article and it should be written in the first person.
  - Don't describe yourself.
  - Don't refer to yourself in your article, for example, don't say: "As a snarky 30 something New Yorker, etc..."
  - Put names and numbers in bold, like this: **John Smith** or **$1,500**
  - 400 words or less for the Scoop.

  ### Sources
  1. Ordered list
  2. of the 10 most relevant sources and urls from RESEARCH
    """

    input_vars = ['breakdown', 'research']
    article_draft_prompt = PromptTemplate(
        input_variables=input_vars, template=_article_draft_prompt
    )

    return LLMChain(llm=llm, prompt=article_draft_prompt, verbose=True)


def get_yt_url(query):
  vid_search = VideosSearch(query, limit = 3)

  return vid_search.result()['result'][0]['link']


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




def main(draft_article, article_name):
  # Parse Articles / Sections
  article_parser = ArticleParser()
  articles = article_parser.parse(draft_article)

  # Init LLM and Chains
  llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.8)
  #library_retriever = get_library_retriever()
  breakdown_chain = get_breakdown_chain(llm)
  article_chain = get_article_chain(llm)

  # Init local output directories
  output_dir = f'chatterboxoffice/{article_name}'

  os.makedirs(output_dir, exist_ok=True)
  # Loop through each section
  for idx, _article in enumerate(articles.steps):
    article = _article.value
    print(article)

    # Get questions for this section
    breakdown = breakdown_chain({'input': article})
    breakdown = breakdown['text']
    
    if "Movie or TV Series:" in breakdown:
      yt_query = breakdown.split("Movie or TV Series:")[1].split("Research Questions")[0].strip()
      yt_url = get_yt_url(yt_query)
    else:
      yt_url = None

    print(f"yt_url: {yt_url}")

    questions = parse_queries(breakdown)

    with open(f'{output_dir}/breakdown.txt', 'w', encoding="utf-8") as f:
      f.write(breakdown)

    # Get all research urls
    urls_to_scrape = []
    queries = []
    for _query in questions:
      query = _query.strip()[3:]
      print(query)

      # top n search context
      try:
        top_n_search_results = get_top_n_search(query, 3)
      except:
        try:
          time.sleep(3)
          top_n_search_results = get_top_n_search(query, 3)
        except:
          continue

      # Collect each url for this query
      for _url in top_n_search_results:
        url = _url['link']
        urls_to_scrape.append(url)
        queries.append(query)

    # Scrape and Research all URLs concurrently
    sqs = 'research{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
    queue = SQS(sqs)
    for url, query in zip(urls_to_scrape, queries):
      cloud_research(url, sqs, query)

    # wait for and collect scrape results from SQS
    research_context = queue.collect(len(urls_to_scrape), max_wait=600)
    research = '\n'.join(research_context)
    
    _prompt = article_chain.prompt.format_prompt(**{
      'breakdown': breakdown,
      'research': research
    })

    # Write each section
    res = article_chain({
      'breakdown': breakdown,
      'research': research
    })

    article = res['text']

    # insert yt video, if there is one
    if yt_url:
      yt_insert = f"[youtube {yt_url}]"
      article = article.replace("[YT_PLACEHOLDER - DON'T REMOVE]", yt_insert)
    else:
      article = article.replace("### Video\n[YT_PLACEHOLDER - DON'T REMOVE]\n", "")      

    # Post to WordPress
    try:
      res = create_wp_post(article)
    except:
      print('Did not post to WordPress.')

    try:
      with open(f'{output_dir}/prompt.txt', 'w', encoding="utf-8") as f:
          #f.write(f'{heading}\n')
          f.write(_prompt.text)

      with open(f'{output_dir}/article.txt', 'w', encoding="utf-8") as f:
          #f.write(f'\n\n{heading}\n')
          f.write(article)
          f.write('\n\n')
    except:
      print("Not everything saved correctly.")

    # Upload to S3 (and convert to JSON)
    article_json = {
      'title': article.split('\n')[0].replace('#', '').strip(),
      'content': "\n".join(article.split('\n')[1:]).strip(),
      'date': datetime.now().astimezone().isoformat(),
      'author': "chatterboxofficestaff",
      'source': "ChatterBoxOffice",
      'url': res['URL']
    }

    with open(f'{output_dir}/article.json', 'w', encoding="utf-8") as file:
      json.dump(article_json, file)

    try:
      s3_client = boto3.client('s3')

      print(os.listdir(output_dir))

      for fname in [f for f in os.listdir(output_dir) if f.endswith('.txt') or f.endswith('.json')]:
        _ = s3_client.upload_file(f'{output_dir}/{fname}', BUCKET, f'{output_dir}/{fname}')
    except:
      print('Skipping S3 upload in local mode')

    



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--article_url', default=None, type=str)
  parser.add_argument('--article_path', default=None, type=str)
  parser.add_argument('--article', default=None, type=str)
  parser.add_argument('--article_name', default=None, type=str)
  args, _ = parser.parse_known_args()
  print(args.article_url)
  print(args.article_path)
  print(args.article)
  print(args.article_name)


  if args.article_url:
    ew_scraper = EWScraper()
    result = ew_scraper.scrape_post(args.article_url)

    article_name = result['title'][:30].replace(' ', '_').replace(':', '_')
    article = 'ARTICLE:\n' + result['content']

  if args.article_path:
    article_name = pathlib.Path(args.article_path).name.replace('.txt', '')
    with open(args.article_path, encoding="utf-8") as f:
      article = f.read()

  if args.article and args.article_name:
    article = args.article
    article_name = args.article_name
    
  main(article, article_name)

  