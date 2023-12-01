import sys
sys.path.append('..')

try:
  from dotenv import load_dotenv
  load_dotenv()
except:
  pass

import os
import json
import boto3
import weaviate as wv
from datetime import datetime

from utils.response_lib import *
from utils.weaviate_utils import get_wv_class_name, create_library, delete_library
from utils.bubble import get_bubble_doc
from utils.cloud_funcs import cloud_scrape


BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
  LAMBDA_DATA_DIR = '.'
else:
  LAMBDA_DATA_DIR = '/tmp'


def to_json_doc(doc_name, doc_content, url=""):
    doc_json = {
        'title': doc_name,
        'content': doc_content,
        'date': datetime.now().astimezone().isoformat(),
        'author': "",
        'source': doc_name,
        'url': url
    }

    local_doc_path = f'{LAMBDA_DATA_DIR}/{doc_name}.json'
    upload_suffix = 'json'
    with open(local_doc_path, 'w', encoding="utf-8") as file:
        json.dump(doc_json, file)

    return local_doc_path, upload_suffix


# Lambda Handler
def data_class(event, context):
    try:
        action = event['body']['action']
        email = event['body']['email']
        name = event['body']['name']
    except:
        action = json.loads(event['body'])['action']
        email = json.loads(event['body'])['email']
        name = json.loads(event['body'])['name']
       
    cls_name, account_name = get_wv_class_name(email, name)

    if action == 'create_library':
       create_library(cls_name)
       return success({'class_name': cls_name})

    if action == 'delete_library':
       delete_library(cls_name)
       return success({'class_name': cls_name})
    

# Lambda Handler
def upload_to_s3(event, context):
  try:
      email = event['body']['email']
      name = event['body']['name']
      doc_url = event['body']['doc_url']
  except:
      email = json.loads(event['body'])['email']
      name = json.loads(event['body'])['name']
      doc_url = json.loads(event['body'])['doc_url']

  # Fetch and write content to local disk
  if 'foxscript.ai' not in doc_url and 'fileupload' not in doc_url:
      doc_file_name = 'scraping.http'
      
      res = cloud_scrape(doc_url, sqs=None, invocation_type='RequestResponse', chunk_overlap=0)
      res_body = json.loads(res['Payload'].read().decode("utf-8"))
      content = json.loads(res_body['body'])['chunks'].replace('<SPLIT>', ' ')

      doc_name = doc_url.replace('https://','').replace('http://','').replace('/', '_')
  else:
      doc_file_name = doc_url.split('/')[-1]
      doc_name = doc_url.split('/')[-1].replace('.txt','').replace('.pdf','')

      # download new document from bubble
      bubble_doc_path = f'{LAMBDA_DATA_DIR}/{doc_file_name}'
      get_bubble_doc(doc_url, bubble_doc_path)
  

  # Prep for upload to S3 (and convert to JSON)
  if doc_file_name.endswith('.http'):
      local_doc_path, upload_suffix = to_json_doc(doc_name, content, url=doc_url)

  if doc_file_name.endswith('.txt'):
      with open(bubble_doc_path, 'r', encoding="utf-8") as f:
          content = f.read()

      local_doc_path, upload_suffix = to_json_doc(doc_name, content)
  
  if doc_file_name.endswith('.pdf'):
      local_doc_path = bubble_doc_path
      upload_suffix = 'pdf'


  print(os.listdir(LAMBDA_DATA_DIR))

  s3_client = boto3.client('s3')
  
  cls_name, account_name = get_wv_class_name(email, name)
  doc_s3_key = f'{account_name}/{cls_name}/{doc_name}.{upload_suffix}'
  _ = s3_client.upload_file(local_doc_path, BUCKET, doc_s3_key)

  try:
      os.remove(local_doc_path)
      print(f"{local_doc_path} removed")

      os.remove(bubble_doc_path)
      print(f"{bubble_doc_path} removed")
  except:
      pass
  
  return success({'s3_key': doc_s3_key})