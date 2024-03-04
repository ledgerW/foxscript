import os
import requests


BUBBLE_API_KEY = os.getenv('BUBBLE_API_KEY')
BUBBLE_API_ROOT = os.getenv('BUBBLE_API_ROOT')
BUBBLE_WF_API_ROOT = os.getenv('BUBBLE_WF_API_ROOT')

if os.getenv('IS_OFFLINE'):
  LAMBDA_DATA_DIR = '.'
else:
  LAMBDA_DATA_DIR = '/tmp'


def upload_bubble_file(path):
    if path.endswith('.pdf'):
        file = {path.split('/')[-1]: (path.split('/')[-1], open(path, 'rb'), 'application/pdf')}
    else:
        file = {'document': open(path,'rb')}

    # https://app.foxscript.ai/version-test/fileupload
    endpoint = BUBBLE_API_ROOT.split('api')[0] + 'fileupload'

    res = requests.post(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
        files=file
    )
    file_id = res.text.split('bubble.io')[-1].replace('"','')
    bubble_url = endpoint + file_id
    
    return bubble_url


def get_bubble_doc(url, local_doc_path):
    response = requests.get(url, headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'})
    if response.status_code != 200:
        print('problem')

    # Save the file to /tmp/ directory
    with open(local_doc_path, 'wb') as f:
        f.write(response.content)


def create_bubble_object(table, body):
    endpoint = BUBBLE_API_ROOT + f'/{table}'

    res = requests.post(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
        json=body
    )

    return res


def get_bubble_object(table, uid):
    endpoint = BUBBLE_API_ROOT + f'/{table}' + f'/{uid}'

    res = requests.get(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
    )

    return res


def delete_bubble_object(table, uid):
    endpoint = BUBBLE_API_ROOT + f'/{table}' + f'/{uid}'

    res = requests.delete(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
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