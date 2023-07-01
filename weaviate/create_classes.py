import weaviate as wv
import argparse

from dotenv import dotenv_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='local', type=str) 
    args, _ = parser.parse_known_args()
    
    # Connect to Weaviate
    config = dotenv_values(f'.env.{args.stage}')

    print('Connecting to {}'.format(config['WEAVIATE_URL']))

    auth_config = wv.auth.AuthApiKey(api_key=config['WEAVIATE_API_KEY'])
    wv_client = wv.Client(url=config['WEAVIATE_URL'], auth_client_secret=auth_config)
    schema = wv_client.schema.get()
    print([cl['class'] for cl in schema['classes']])
    
    # DataSource
    try:
      wv_client.schema.create_class('weaviate/schema/DataSource.json')
    except Exception as e:
      if 'already used' in e.message:
        print('SKIPPING - Class already exists')
    
    # ThreatIntelReport
    try:
      try:
        wv_client.schema.create_class('weaviate/schema/ThreatIntelReport.json')
      except:
        print('ThreatIntelChunk class not created yet - will create')

      wv_client.schema.create_class('weaviate/schema/ThreatIntelChunk.json')
      wv_client.schema.delete_class('ThreatIntelReport')
      wv_client.schema.create_class('weaviate/schema/ThreatIntelReport.json')
    except Exception as e:
      if 'already used' in e.message:
        print('SKIPPING - Class already exists')


    # User and Conversation and Folder
    try:
      try:
        wv_client.schema.create_class('weaviate/schema/User.json')
      except:
        print('User class not created yet - will create')

      wv_client.schema.create_class('weaviate/schema/Conversation.json')
      wv_client.schema.create_class('weaviate/schema/Folder.json')
      wv_client.schema.create_class('weaviate/schema/Prompt.json')
      wv_client.schema.delete_class('User')
      wv_client.schema.create_class('weaviate/schema/User.json')
    except Exception as e:
      if 'already used' in e.message:
        print('SKIPPING - Class already exists')


    schema = wv_client.schema.get()
    print([cl['class'] for cl in schema['classes']])