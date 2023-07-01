import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()

#WP_API_KEY = os.environ['WP_API_KEY']

import os
import argparse
import time
import requests
import pathlib
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

from utils.scrapers.base import Scraper
from utils.scrapers import EWScraper
from utils.content import handle_pdf
from utils.MemoryRetrievalChain import MemoryRetrievalChain

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")


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
    breakdown_template = """You are on the marketing team for Mosaic, a technology consulting group that focuses on machine learning, AI, and data analytic solutions for commerical and government clients.

  To create marketing blog content, Mosaic starts with the subject of an existing ARTICLE, then does more research on the subject, 
  and then writes an educational article identifying practical applications for potential customers.

  Our research focuses on:
  1. discovering the broader context of the concepts in the ARTICLE by learning what led to them, what else is happening now,
  and where might they lead in the future.
  2. quantifying trends described in the ARTICLE
  3. estimating market size or financial opportunity of cost of what's described in the ARTICLE
  4. problems businesses and organizations may face related to this ARTICLE
  5. potential solutions businesses and organizations can pursue related to this ARTICLE
  6. ML and AI technologies and services currently available related to this ARTICLE
  7. Open source libraries (preferably in Python) related to this ARTICLE
  8. Research papers related to this ARTICLE
  9. Opinions of other subject matter experts related to this ARTICLE
  10. Real world case studies or examples of what's described in the ARTICLE

  Given the ARTICLE below...
  ARTICLE:
  {input}

  Perform the following tasks in the following format:

  Research Questions:
  1. ordered list of
  2. 10-12 questions we want to research related to the ARTICLE and our research focus above.
  
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
    _article_draft_prompt = """You are writing an article for Mosaic,
  a technology consulting group that focuses on machine learning, AI, and data analytic solutions for commerical and government clients.
  The article is backed by research.
  The article should use specific names and numbers.
  Important - the article should use markdown footnotes throughout to cite sources from RESEARCH.

  You are a senior marketing professional and technology subject matter expert.
  You are curious and friendly, and you aim to educate the reader in a casual tone.
  DO NOT refer to yourself in your writing.
  DO NOT start sentences with "As a senior marketing professional..." or "as a curious person...", etc... or use that pattern at all!

  Your job is to use the QUESTIONS and RESEARCH below to write a well researched article.

  If the RESEARCH seems unrelated to what's in the QUESTIONS, you can ignore it.

  QUESTIONS:
  {breakdown}

  RESEARCH:
  {research}

  You get bonus points if you can accomplish the following in your article:
  - Important - the article should use markdown footnotes throughout to cite sources from RESEARCH.
  - Create a markdown table to display information

  Write the article using markdown according to the following pattern:

  # Clever Title With An Emoji Here

  ### TLDR
  - Bullet Point List
  - of Key Points of the article
  - with markdown footnotes throughout to cite sources from RESEARCH

  ### Thoughts
  - This is the main body of the article and it should be written in the first person.
  - Don't describe yourself.
  - Don't refer to yourself in your article, for example, don't say: "As a curious professional, etc..."
  - the article should use markdown footnotes throughout to cite sources from RESEARCH.
  - Put names and numbers in markdown bold, like this: **John Smith** or **$1,500**
  - Compare some information in a markdown table, so it's easy for the reader to visualize
  - 500 words or less for the Thoughts.

  ### Case Study
  Is there a real example case study in the RESEARCH?
  If so, describe it here, but if not, just omit the entire Case Study section.

  ### Sources
  1. Ordered list
  2. of sources cited by footnote above
    """

    input_vars = ['breakdown', 'research']
    article_draft_prompt = PromptTemplate(
        input_variables=input_vars, template=_article_draft_prompt
    )

    return LLMChain(llm=llm, prompt=article_draft_prompt, verbose=True)


#def get_yt_url(query):
#  vid_search = VideosSearch(query, limit = 3)

#  return vid_search.result()['result'][0]['link']


def get_article_img(article):
  key_img_1 = article.split('### TLDR')[1].split('\n')[1]

  search = GoogleSerperAPIWrapper(type="images")
  results = search.results(key_img_1)

  img_url = results['images'][0]['imageUrl']
  for img in results['images']:
    if (img['imageWidth'] > img['imageHeight']) and ('.jpg' in img['imageUrl']):
        img_url = img['imageUrl']
        break
    
  return img_url


#def upload_wp_media_image(img_url):
#    url = 'https://public-api.wordpress.com/rest/v1.1/sites/chatterboxoffice.com/media/new'
#    headers = {
#        'Authorization': f'Bearer {WP_API_KEY}',  # Replace with your actual API token
#    }
#    data = {
#        'media_urls': img_url
#    }

#    response = requests.post(url, headers=headers, data=data)
#    response.raise_for_status()  # Raise exception if the request failed

#    return response.json()['media'][0]['ID']


#def create_wp_post(article):
#    try:
#      img_url = get_article_img(article)
#      img_id = upload_wp_media_image(img_url)
#    except:
#      img_url = "https://chatterboxoffice.com/wp-content/uploads/2023/06/cropped-transparent-logo.png"
#      img_id = upload_wp_media_image(img_url)

#    post_url = 'https://public-api.wordpress.com/rest/v1.1/sites/chatterboxoffice.com/posts/new'  # Replace with your actual site ID
#    post_headers = {
#        'Authorization': f'Bearer {WP_API_KEY}',  # Replace with your actual API token
#    }
#    post_data = {
#        'title': article.split('\n')[0].replace('#', '').strip(),
#        'content': "\n".join(article.split('\n')[1:]).strip(),
#        'featured_image': img_id,
#        'status': 'draft',
#        'author': 'chatterboxofficestaff',
#        'format': 'standard'
#    }

#    post_response = requests.post(post_url, headers=post_headers, data=post_data)

#    return post_response.json()




def main(draft_article, article_name):
  llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.8)
  
  article_parser = ArticleParser()
  articles = article_parser.parse(draft_article)

  #library_retriever = get_library_retriever()
  breakdown_chain = get_breakdown_chain(llm)
  article_chain = get_article_chain(llm)

  os.makedirs(f'mosaic/{article_name}', exist_ok=True)
  # Loop through each section
  for idx, _article in enumerate(articles.steps):
    article = _article.value
    print(article)

    # Get questions for this section
    breakdown = breakdown_chain({'input': article})
    breakdown = breakdown['text']
    
    #if "Movie or TV Series:" in breakdown:
    #  yt_query = breakdown.split("Movie or TV Series:")[1].split("Research Questions")[0].strip()
    #  yt_url = get_yt_url(yt_query)
    #else:
    #  yt_url = None

    #print(f"yt_url: {yt_url}")

    questions = parse_queries(breakdown)

    with open(f'mosaic/{article_name}/breakdown.txt', 'w', encoding="utf-8") as f:
      f.write(breakdown)

    research_context = []
    #library_context = []
    # Loop through questions
    for _query in questions:
      query = _query.strip()[3:]
      print(query)

      # library context
      #library_query_context = get_context(query, llm, library_retriever)
      #library_context.append(library_query_context)

      # top n search context
      try:
        top_n_search_results = get_top_n_search(query, 3)
      except:
        try:
          time.sleep(3)
          top_n_search_results = get_top_n_search(query, 3)
        except:
          continue

      # Look through top n urls
      for _url in top_n_search_results:
        try:
          url = _url['link']
          print(url)

          chunks = scrape_and_chunk(url, 100, tokenizer)
          vec_db = get_ephemeral_vecdb(chunks, {'source': url})
          src_context = get_context(query, llm, vec_db.as_retriever())
          research_context.append(f"query: {query}\n" + f"source: {url}\n" + src_context + '\n')
        except:
          print('Issue with {}'.format(url))
          continue
      
    #full_library = '\n'.join(library_context)
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

    try:
      img_url = get_article_img(article)
    except:
      print('No img url')
      print(img_url)

    # insert yt video, if there is one
    #if yt_url:
    #  yt_insert = f"[youtube {yt_url}]"
    #  article = article.replace("[YT_PLACEHOLDER - DON'T REMOVE]", yt_insert)
    #else:
    #  article = article.replace("### Video\n[YT_PLACEHOLDER - DON'T REMOVE]\n", "")      

    try:
      with open(f'mosaic/{article_name}/prompt.txt', 'w', encoding="utf-8") as f:
          #f.write(f'{heading}\n')
          f.write(_prompt.text)

      with open(f'mosaic/{article_name}/article.txt', 'w', encoding="utf-8") as f:
          f.write('Potential article header image: {}\n\n'.format(img_url))
          f.write(article)
          f.write('\n\n')
    except:
      print("Not everything saved correctly.")

    #try:
    #  res = create_wp_post(article)
    #except:
    #  print('Did not post to WordPress.')

    



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--article_url', default=None, type=str)
  parser.add_argument('--article_path', default=None, type=str)
  args, _ = parser.parse_known_args()
  print(args.article_url)
  print(args.article_path)


  if args.article_url:
    ew_scraper = EWScraper()
    result = ew_scraper.scrape_post(args.article_url)

    article_name = result['title'][:30].replace(' ', '_').replace(':', '_')
    article = 'ARTICLE:\n' + result['content']

  if args.article_path:
    article_name = pathlib.Path(args.article_path).name.replace('.txt', '')
    with open(args.article_path, encoding="utf-8") as f:
      article = f.read()
    
  main(article, article_name)

  