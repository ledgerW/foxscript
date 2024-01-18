import os
import json
import urllib
import boto3
import requests
import time
from docx import Document

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

s3_client = boto3.client('s3')


# Helpers
def get_docx_text(file_name):
    doc = Document(file_name)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    return '\n'.join(full_text)


def split_text(doc_text_or_parts, max_word_len=3000):
    """
    Primarily for Originality.ai plagiarism checker API.
    They have a 3000 word limit. In case our document is 
    greater than the limit, we will split it in half
    recursively until all parts are less than the limit.
    """
    if type(doc_text_or_parts) == list:
        doc_parts = doc_text_or_parts
    else:
        doc_parts = [doc_text_or_parts]

    part_len = len(doc_parts[0].split(' '))
    split_len = int(part_len/2)
    
    if part_len >= max_word_len:
        new_doc_parts = []
        for i, part in enumerate(doc_parts):
            new_doc_parts.append(' '.join(part.split(' ')[:split_len]))
            new_doc_parts.append(' '.join(part.split(' ')[split_len:]))

        return split_text(new_doc_parts, max_word_len)
    else:
        return doc_parts
    

def get_plagiarism_report(content_text, title, customer_url):
    endpoint = "https://api.originality.ai/api/v1/scan/plag"

    headers = {
      'Accept': 'application/json',
      'X-OAI-API-KEY': os.getenv('ORIGINALITY_PLAG')
    }
    print(headers)

    payload = {
      "content": content_text,
      "title": title,
      "excludedUrl": customer_url
    }

    print(payload)

    res = requests.post(endpoint, headers=headers, json=payload)

    return res


# Lambda Handler
def handler(event, context):
    """
    Expected S3 Key pattern: customer_url/batch_id/document.suffix
    """
    print(event)
    if 'Records' in event:
        # S3 Trigger
        bucket = urllib.parse.unquote_plus(event['Records'][0]['s3']['bucket']['name'], encoding='utf-8')
        doc_key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

        customer_url = doc_key.split('/')[-3]
        batch_id = doc_key.split('/')[-2]
        doc_file_name = doc_key.split('/')[-1]
        local_doc_path = LAMBDA_DATA_DIR + f'/{doc_file_name}'
        s3_client.download_file(bucket, doc_key, local_doc_path)
    else:
        # HTTP Trigger
        try:
            bucket = event['body']['bucket']
            body = event['body']
        except:
            bucket = json.loads(event['body'])['bucket']
            body = json.loads(event['body'])

    # read document in .docx format (default google doc download format)
    if local_doc_path.endswith('.docx'):
        doc_text = get_docx_text(local_doc_path)
    else:
        with open(local_doc_path, "r") as f:
            doc_text = f.read()
    print("Document has {} words".format(len(doc_text.split(' '))))

    # Split into parts if greater than plag checker API word limit
    doc_parts = split_text(doc_text)
    print("Document split into {} parts".format(len(doc_parts)))

    # Send each part to plag checker
    for part_num, part in enumerate(doc_parts):
        title = f"{part_num+1}_{doc_file_name.replace('.docx','').replace('.txt','').replace('.md','')}"
        print(f"Getting report for {title}")
        res = get_plagiarism_report(part, title, customer_url)
        
        local_report_file_name = f'{title}.json'
        local_report_path = LAMBDA_DATA_DIR + f'/{local_report_file_name}'
        with open(local_report_path, 'w') as f:
            f.write(json.dumps(res.json()))

        s3_report_key = f"{customer_url}/{batch_id}/plag_reports/{local_report_file_name}"
        print(f"Uploading {s3_report_key} to S3")
        _ = s3_client.upload_file(local_report_path, bucket, s3_report_key)
        time.sleep(2)


