import sys
sys.path.append('..')

import os
import json
import pandas as pd
from utils.bubble import (
    create_bubble_object,
    get_bubble_object,
    update_bubble_object,
    get_bubble_doc,
    delete_bubble_object,
    upload_bubble_file
)
from utils.response_lib import *

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
  LAMBDA_DATA_DIR = '.'
else:
  LAMBDA_DATA_DIR = '/tmp'



# Lambda Handler
def handler(event, context):
    """
    """
    print(event)

    try:
       body = json.loads(event['body'])
    except:
       body = event['body']

    # Fetch batch input file from bubble
    ecs_doc_id = body['ecs_doc_id']
    res = get_bubble_object('ecs-doc', ecs_doc_id)
    ecs_doc_object = res.json()['response']

    keyword_doc_url = ecs_doc_object['url']
    keyword_doc_file_name = keyword_doc_url.split('/')[-1]
    local_keyword_doc_path = f'{LAMBDA_DATA_DIR}/{keyword_doc_file_name}'

    if 'app.foxscript.ai' in keyword_doc_url:
        get_bubble_doc(keyword_doc_url, local_keyword_doc_path)
        print("Retrieved keyword doc from bubble")
    else:
        local_batch_path = keyword_doc_url
        print("Using local batch file")

    # Get Topics and Stats
    keywords_df = pd.read_csv(local_keyword_doc_path)

    n_topics = keywords_df.shape[0]
    has_keyword_col = 'Keyword' in keywords_df.columns
    has_volume_col = ('Search Volume' in keywords_df.columns) or ('Volume' in keywords_df.columns)

    # Update ECS Doc Object in Bubble
    object_body = {
       'n_topics': n_topics,
       'has_keyword_col': has_keyword_col,
       'has_volume_col': has_volume_col
    }
    _ = update_bubble_object('ecs-doc', ecs_doc_id, object_body)

    return success({'success': True})
    



