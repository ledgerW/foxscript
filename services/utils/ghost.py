try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import os
import json
import requests
from datetime import datetime as date

from utils.cloud_funcs import cloud_get_secret


if os.getenv('IS_OFFLINE') or not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
   LAMBDA_DATA_DIR = '.'
else:
   LAMBDA_DATA_DIR = '/tmp'



def get_ghost_jwt(user_id: str, api: str) -> dict:
    import jwt	# pip install pyjwt

    res = cloud_get_secret(api, user_id)
    key = res['SecretString']

    if api == 'ghost-content':
        return key

    # Split the key into ID and SECRET
    id, secret = key.split(':')
    
    # Prepare header and payload
    iat = int(date.now().timestamp())

    header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
    payload = {
        'iat': iat,
        'exp': iat + 5 * 60,
        'aud': '/admin/'
    }

    # Create the token (including decoding secret)
    token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)

    return token


def get_article_img(title: str, download: bool=False):
    from langchain_community.utilities import GoogleSerperAPIWrapper

    def get_img_url(results):
        img_url = results['images'][0]['imageUrl']
        for img in results['images']:
            if (img['imageWidth'] > img['imageHeight']) and (img['imageUrl'].endswith('.jpg')):
                img_url = img['imageUrl']
                return img_url
        return img_url

    search = GoogleSerperAPIWrapper(type="images")
    results = search.results(title)

    img_url = get_img_url(results)

    local_img_path = None
    if download:
        try:
            response = requests.get(img_url)
            local_img_path = os.path.join(LAMBDA_DATA_DIR, img_url.split('/')[-1])
        except:
            try:
                new_results = {}
                new_results['images'] = [img for img in results['images'] if img['imageUrl'] != img_url]
                img_url = get_img_url(new_results)
                response = requests.get(img_url)
                local_img_path = os.path.join(LAMBDA_DATA_DIR, img_url.split('/')[-1])
            except:
                pass
        
        if local_img_path:
            with open(local_img_path, "wb") as file:
                file.write(response.content)

    return local_img_path if local_img_path else img_url


def build_body(title: str, content: str, tags: list[str], author_email: str, img_path: str=None, status: str='draft') -> dict:
    lexical = {
        'root': {
            'children': [
                {'type': 'markdown', 'version': 1, 'markdown': ''},
                {
                    'children': [],
                    'direction': None,
                    'format': '',
                    'indent': 0,
                    'type': 'paragraph',
                    'version': 1
                }
            ],
            'direction': None,
            'format': '',
            'indent': 0,
            'type': 'root',
            'version': 1
        }
    }
    lexical['root']['children'][0]['markdown'] = content

    body = {
        'posts': [{
            'title': title,
            'slug': title.lower().replace(' ', '-').replace(':', '-'),
            'lexical': json.dumps(lexical),
            'status': status,
            'tags': tags,
            'authors': [author_email]
        }]
    }

    if img_path:
        body['posts'][0]['feature_image'] = img_path
        
    return body


def call_ghost(user_id: str, domain: str, endpoint_type: str, body: dict=None, img_path: str=None):
    ENDPOINTS = {
        'post': {
            'endpoint': '/posts/?formats={}',
            'api': 'ghost-admin'
        },
        'image': {
            'endpoint': '/images/upload/',
            'api': 'ghost-admin'
        },
        'site': {
            'endpoint': '/site/',
            'api': 'ghost-admin'
        },
        'tags': {
            'endpoint': '/tags/',
            'api': 'ghost-content'
        }
    }
    jwtoken = get_ghost_jwt(user_id, ENDPOINTS[endpoint_type]['api'])
    headers = {'Authorization': 'Ghost {}'.format(jwtoken)}

    # Make an authenticated request to create a post
    if endpoint_type == 'post':
        format_type = 'html' if 'html' in body['posts'][0] else 'lexical'
        endpoint = ENDPOINTS[endpoint_type]['endpoint'].format(format_type)
        url = f'https://{domain}/ghost/api/admin' + endpoint
        res = requests.post(url, headers=headers, json=body)

    if endpoint_type == 'image':
        with open(img_path, 'rb') as img_file:
            files = {'file': (img_path.split('/')[-1], img_file, 'image/jpeg')}
            endpoint = ENDPOINTS[endpoint_type]['endpoint']
            url = f'https://{domain}/ghost/api/admin' + endpoint
            res = requests.post(url, headers=headers, files=files)

    if endpoint_type == 'tags':
        endpoint = ENDPOINTS[endpoint_type]['endpoint']
        url = f'https://{domain}/ghost/api/content' + endpoint + f'?key={jwtoken}&limit=all'
        res = requests.get(url)

    return res.json()