import sys
sys.path.append('..')

from dotenv import load_dotenv
load_dotenv()

import os
import json
import requests
import boto3
import weaviate as wv
from datetime import datetime

from utils.workflow import get_wv_class_name
from utils.response_lib import *

BUBBLE_API_KEY = os.getenv('BUBBLE_API_KEY')
BUCKET = os.getenv('BUCKET')


LAMBDA_DATA_DIR = '/tmp'

auth_config = wv.auth.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])

wv_client = wv.Client(
    url=os.environ['WEAVIATE_URL'],
    additional_headers={
        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY']
    },
    auth_client_secret=auth_config
)


def create_library(name):
    content_cls_name = "{}Content".format(name)
    chunk_cls_name = "{}Chunk".format(name)

    content_ref_prop = {
      'name': 'hasChunks',
      'dataType': [chunk_cls_name],
      'description': 'Chunks of content'
    }
    chunk_ref_prop = {
      'name': 'fromContent',
      'dataType': [content_cls_name],
      'description': 'Original content'
    }

    # make new source content class
    with open('weaviate/schema/Content.json', 'r', encoding='utf-8') as f:
      content_cls = json.loads(f.read())

    content_cls['class'] = content_cls_name
    props = [prop for prop in content_cls['properties'] if prop['name'] != 'hasChunks'] + [content_ref_prop]
    content_cls['properties'] = props


    # make new source chunk class
    with open('weaviate/schema/Chunk.json', 'r', encoding='utf-8') as f:
      chunk_cls = json.loads(f.read())

    chunk_cls['class'] = chunk_cls_name
    props = [prop for prop in chunk_cls['properties'] if prop['name'] != 'fromContent'] + [chunk_ref_prop]
    chunk_cls['properties'] = props

    try:
      try:
        wv_client.schema.create_class(content_cls)
      except:
        print(f'{chunk_cls_name} class not created yet - will create')

      wv_client.schema.create_class(chunk_cls)
      wv_client.schema.delete_class(content_cls_name)
      wv_client.schema.create_class(content_cls)
    except Exception as e:
      if 'already used' in e.message:
        print('SKIPPING - Class already exists')


    schema = wv_client.schema.get()
    print([cl['class'] for cl in schema['classes']])


def delete_library(name):
    content_cls_name = "{}Content".format(name)
    chunk_cls_name = "{}Chunk".format(name)

    wv_client.schema.delete_class(content_cls_name)
    wv_client.schema.delete_class(chunk_cls_name)

    schema = wv_client.schema.get()
    print([cl['class'] for cl in schema['classes']])


def get_bubble_doc(url, local_doc_path):
    response = requests.get(url, headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'})
    if response.status_code != 200:
        print('problem')

    # Save the file to /tmp/ directory
    with open(local_doc_path, 'wb') as f:
        f.write(response.content)


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

  doc_file_name = doc_url.split('/')[-1]
  doc_name = doc_url.split('/')[-1].replace('.txt','').replace('.pdf','')

  # download new document from bubble
  bubble_doc_path = f'{LAMBDA_DATA_DIR}/{doc_file_name}'
  get_bubble_doc(doc_url, bubble_doc_path)
  
  cls_name, account_name = get_wv_class_name(email, name)

  # Prep for upload to S3 (and convert to JSON)
  if doc_file_name.endswith('.txt'):
    with open(bubble_doc_path, 'r', encoding="utf-8") as f:
      bubble_doc = f.read()

    doc_json = {
      'title': doc_name,
      'content': bubble_doc,
      'date': datetime.now().astimezone().isoformat(),
      'author': "",
      'source': "",
      'url': ""
    }

    local_doc_path = f'{LAMBDA_DATA_DIR}/{doc_name}.json'
    upload_suffix = 'json'
    with open(local_doc_path, 'w', encoding="utf-8") as file:
      json.dump(doc_json, file)
  
  if doc_file_name.endswith('.pdf'):
     local_doc_path = bubble_doc_path
     upload_suffix = 'pdf'


  print(os.listdir(LAMBDA_DATA_DIR))

  s3_client = boto3.client('s3')
  
  doc_s3_key = f'{account_name}/{cls_name}/{doc_name}.{upload_suffix}'
  _ = s3_client.upload_file(local_doc_path, BUCKET, doc_s3_key)
  
  return success({'s3_key': doc_s3_key})