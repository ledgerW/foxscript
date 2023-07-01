import sys
sys.path.append('..')

import os
import json
import weaviate as wv

from utils.response_lib import *


lambda_data_dir = '/tmp'

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
    with open('Content.json', 'r', encoding='utf-8') as f:
      content_cls = json.loads(f.read())

    content_cls['class'] = content_cls_name
    props = [prop for prop in content_cls['properties'] if prop['name'] != 'hasChunks'] + [content_ref_prop]
    content_cls['properties'] = props


    # make new source chunk class
    with open('Chunk.json', 'r', encoding='utf-8') as f:
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


def get_wv_class_name(email, name):
    domain = email.split('@')[1].split('.')[0]
    username = email.split('@')[0].replace('.', '').capitalize()
    account = f"{username}{domain}"
    name = name.capitalize()

    return f"{account}{name}"



def data_class(event, context):
    try:
        action = event['body']['action']
        email = event['body']['email']
        name = event['body']['name']
    except:
        action = json.loads(event['body'])['action']
        email = json.loads(event['body'])['email']
        name = json.loads(event['body'])['name']
       
    cls_name = get_wv_class_name(email, name)

    if action == 'create_library':
       create_library(cls_name)
       return success({'class_name': cls_name})

    if action == 'delete_library':
       delete_library(cls_name)
       return success({'class_name': cls_name})