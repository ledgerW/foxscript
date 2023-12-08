import os
import json
from datetime import datetime

import weaviate as wv

if os.getenv('IS_OFFLINE'):
   LAMBDA_DATA_DIR = '.'
   WEAVIATE_SCHEMA_DIR = '../../'
else:
   LAMBDA_DATA_DIR = '/tmp'
   WEAVIATE_SCHEMA_DIR = ''


wv_client = wv.Client(
    url=os.environ['WEAVIATE_URL'],
    additional_headers={
        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY']
    },
    auth_client_secret=wv.auth.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])
)


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


def get_wv_class_name(email, name):
    domain = email.split('@')[1].split('.')[0]
    username = email.split('@')[0].replace('.', '').capitalize()
    account = f"{username}{domain}"
    name = name.capitalize()

    cls_name = f"{account}{name}"

    return cls_name, account


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
    with open(WEAVIATE_SCHEMA_DIR + 'weaviate/schema/Content.json', 'r', encoding='utf-8') as f:
      content_cls = json.loads(f.read())

    content_cls['class'] = content_cls_name
    props = [prop for prop in content_cls['properties'] if prop['name'] != 'hasChunks'] + [content_ref_prop]
    content_cls['properties'] = props


    # make new source chunk class
    with open(WEAVIATE_SCHEMA_DIR + 'weaviate/schema/Chunk.json', 'r', encoding='utf-8') as f:
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