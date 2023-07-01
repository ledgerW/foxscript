import weaviate as wv
import argparse
import json

from dotenv import dotenv_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='local', type=str)
    parser.add_argument('--source', default='local', type=str) 
    args, _ = parser.parse_known_args()
    
    # Connect to Weaviate
    config = dotenv_values(f'.env.{args.stage}')

    print('Connecting to {}'.format(config['WEAVIATE_URL']))

    auth_config = wv.auth.AuthApiKey(api_key=config['WEAVIATE_API_KEY'])
    wv_client = wv.Client(url=config['WEAVIATE_URL'], auth_client_secret=auth_config)
    schema = wv_client.schema.get()
    print([cl['class'] for cl in schema['classes']])
    
    
    # New Source
    content_cls_name = "{}Content".format(args.source)
    chunk_cls_name = "{}Chunk".format(args.source)

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