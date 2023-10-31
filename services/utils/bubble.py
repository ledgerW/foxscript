import os
import requests


BUBBLE_API_KEY = os.getenv('BUBBLE_API_KEY')
BUBBLE_API_ROOT = os.getenv('BUBBLE_API_ROOT')
BUBBLE_WF_API_ROOT = os.getenv('BUBBLE_WF_API_ROOT')

if os.getenv('IS_OFFLINE'):
  LAMBDA_DATA_DIR = '.'
else:
  LAMBDA_DATA_DIR = '/tmp'


def get_bubble_doc(url, local_doc_path):
    response = requests.get(url, headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'})
    if response.status_code != 200:
        print('problem')

    # Save the file to /tmp/ directory
    with open(local_doc_path, 'wb') as f:
        f.write(response.content)


def write_to_bubble(table, body):
    endpoint = BUBBLE_API_ROOT + f'/{table}'

    res = requests.post(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
        json=body
    )

    return res


def update_bubble_object(table, uid, body):
    endpoint = BUBBLE_API_ROOT + f'/{table}' + f'/{uid}'

    res = requests.patch(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
        json=body
    )

    return res


def trigger_bubble_workflow_api(name, body={}):
    endpoint = BUBBLE_WF_API_ROOT + f'/{name}'

    res = requests.post(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
        json=body
    )

    return res