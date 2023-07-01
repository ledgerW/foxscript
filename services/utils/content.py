import weaviate as wv
import pathlib
import json
import re
from datetime import datetime
import tiktoken
from PyPDF2 import PdfReader
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import sys
sys.path.append('..')

from utils import scrapers
from utils.ScraperConfig import Config


def get_or_create_source(source_config, wv_client):
    where_filter = {
        "path": ["source"],
        "operator": "Equal",
        "valueString": source_config['source']
    }

    result = (
        wv_client.query
        .get("DataSource", "source scraper lastPostDate")
        .with_where(where_filter)
        .do()
    )
    
    if len(result['data']['Get']['DataSource']) == 0:
        data_props = {
            'source': source_config['source'],
            'scraper': source_config['scraper'],
            'lastPostDate': datetime.fromisoformat("2000-01-01T00:00:00").astimezone().isoformat()
        }

        uuid = wv.util.generate_uuid5({'source': source_config['source']}, 'DataSource')
        source_uuid = wv_client.data_object.create(data_props, 'DataSource', uuid=uuid)
        result = data_props
    else:
        result = result['data']['Get']['DataSource'][0]

    return result


def update_source(report_source, latest_post_date, wv_client):
    where_filter = {
        "path": ["source"],
        "operator": "Equal",
        "valueString": report_source
    }

    result = (
        wv_client.query
        .get("DataSource", "_additional{ id }")
        .with_where(where_filter)
        .do()
    )

    wv_client.data_object.update(
        data_object = {'lastPostDate': latest_post_date.isoformat()},
        class_name = 'DataSource',
        uuid = result['data']['Get']['DataSource'][0]['_additional']['id']
    )


def scrape_content(report_source, url=None, wv_client=None):
    config = Config[report_source]
    scraper = getattr(scrapers, config['scraper'])()

    if url: # Scrape specific url/blog (an older one)
        result = scraper.scrape_post(url)
    else: # Check for and scrape the latest and update the date of latest
        source = get_or_create_source(config, wv_client)
        try:
            last_time_scraped = datetime.fromisoformat(source['lastPostDate']) 
        except:
            last_time_scraped = datetime.strptime('2023-05-15T00:00:00Z', "%Y-%m-%dT%H:%M:%SZ").astimezone()

        latest_post_url, latest_post_date = scraper.get_latest_post_meta()

        if latest_post_date > last_time_scraped:
            result = scraper.scrape_post()
            update_source(report_source, latest_post_date, wv_client)
            print('Set new latest date to {}'.format(latest_post_date.isoformat()))
        else:
            result = ''

    return result




def text_splitter(text, n, tokenizer):
    chunks = []
    chunk = ''
    sentences = [s.strip().replace('\n', ' ') for s in text.split('.')]
    for s in sentences:
        # start new chunk
        if chunk == '':
            chunk = s
        else:
            chunk = chunk + ' ' + s
        
        chunk_len = len(tokenizer.encode(chunk))
        if chunk_len >= 0.9*n:
            chunks.append(chunk)
            chunk = ''

    if chunk != '':
        chunks.append(chunk)
    
    return chunks


def multi_page_text_splitter(pages, n, tokenizer):
    chunks = []
    page_nums = []
    chunk = ''
    for num, page in enumerate(pages):
        sentences = [s.strip().replace('\n', ' ') for s in page.split('.')]
        for s in sentences:
            # start new chunk
            if chunk == '':
                chunk = s
            else:
                chunk = chunk + ' ' + s
            
            chunk_len = len(tokenizer.encode(chunk))
            if chunk_len >= 0.9*n:
                chunks.append(chunk)
                page_nums.append(num+1)
                chunk = ''

    if chunk != '':
        chunks.append(chunk)
        page_nums.append(num+1)
    
    return chunks, page_nums


def handle_pdf(report_path, n, tokenizer):
    reader = PdfReader(report_path)
    print(report_path.name)
    print('Found {} pages'.format(len(reader.pages)))

    pages = [p.extract_text() for p in reader.pages]
    chunks, pages = multi_page_text_splitter(pages, n, tokenizer)

    # add metadata to chunk
    if '/CreationDate' in reader.metadata:
        if '+' in reader.metadata['/CreationDate']:
            date = reader.metadata['/CreationDate'].split('+')[0].replace('D:','')
        else:
            date = reader.metadata['/CreationDate'].split('-')[0].replace('D:','')
        try:
            date = datetime.strptime(date, '%Y%m%d%H%M%S').astimezone().isoformat()
        except:
            date = date.split('Z')[0]
            date = datetime.strptime(date, '%Y%m%d%H%M%S').astimezone().isoformat()
        short_date = date.split('T')[0]
    else:
        date = ''

    if '/Author' in reader.metadata:
        author = reader.metadata['/Author']
    else:
        author = ''

    if '/Title' in reader.metadata:
        title = reader.metadata['/Title'].replace(' ', '_')
    else:
        title = ''

    for idx, chunk in enumerate(chunks):
        use_title = title if title != '' else report_path.name
        
        chunk_with_meta = f"""Title: {use_title}
        Author: {author}
        Date: {short_date}
        {chunk}
        """
        
        chunk_with_meta = re.sub(' +', ' ', chunk_with_meta)
        chunks[idx] = chunk_with_meta

    meta = {
        'date': date,
        'url': report_path.name,
        'title': title,
        'author': author,
        'source': ''
    }
        
    return chunks, pages, meta


def handle_json(report_path, n, tokenizer):
    with open(report_path, 'r') as file:
        report_json = json.load(file)
    
    chunks = text_splitter(report_json['content'], n, tokenizer)
    pages = [0 for _ in range(len(chunks))]

    title = report_json['title']
    author = report_json['author']
    date = report_json['date'].split('T')[0]

    for idx, chunk in enumerate(chunks):
        chunk_with_meta = f"""Title: {title}
        Author: {author}
        Date: {date}
        {chunk}
        """

        chunk_with_meta = re.sub(' +', ' ', chunk_with_meta)
        chunks[idx] = chunk_with_meta

    meta = {
        'date': report_json['date'],
        'url': report_json['url'],
        'title': title,
        'author': author,
        'source': report_json['source']
    }

    return chunks, pages, meta


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def load_openai_embedding(data_props, class_name, uuid, wv_client):
    try:
        wv_client.data_object.create(data_props, class_name, uuid)
    except wv.ObjectAlreadyExistsException:
        pass


def load_content(path, source=None, chunk_size=100, wv_client=None):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    report_path = pathlib.Path(path)

    if report_path.name.endswith('.pdf'):
        chunks, pages, meta = handle_pdf(report_path, chunk_size, tokenizer)

        # source NotImplemented Yet for PDF
        content_cls = f'{source}Content'
        chunk_cls = f'{source}Chunk'
    elif report_path.name.endswith('.json'):
        chunks, pages, meta = handle_json(report_path, chunk_size, tokenizer)
        source = meta['source']
        content_cls = f'{source}Content'
        chunk_cls = f'{source}Chunk'

    print('Getting embeddings for {} chunks'.format(len(chunks)))

    if not wv_client:
        return chunks, pages, meta

    # Load into weaviate
    # 1. load all chunks
    # 2. load content
    # 3. add content ref to chunks
    # 4. add chunk refs to content

    # all chunks
    chunk_uuids = []
    for chunk, page in zip(chunks, pages):
        data_props = {
            "chunk": chunk,
            "page": page
        }

        uuid = wv.util.generate_uuid5({'chunk': chunk}, chunk_cls)
        chunk_uuids.append(uuid)

        load_openai_embedding(data_props, chunk_cls, uuid, wv_client)
    print('Chunks loaded')

    # the content
    data_props = meta

    content_uuid = wv.util.generate_uuid5(data_props, content_cls)

    load_openai_embedding(data_props, content_cls, content_uuid, wv_client)
    print('Report loaded')

    # attach report ref to chunks
    for chunk_uuid in chunk_uuids:
        wv_client.batch.add_reference(
            from_object_uuid=chunk_uuid,
            from_object_class_name=chunk_cls,
            from_property_name="fromContent",
            to_object_uuid=content_uuid,
            to_object_class_name=content_cls,
        )
    wv_client.batch.flush()
    print('Content ref attached to Chunks')

    # attach chunk refs to report
    for chunk_uuid in chunk_uuids:
        wv_client.data_object.reference.add(
            from_uuid=content_uuid,
            from_property_name='hasChunks',
            to_uuid=chunk_uuid,
            from_class_name=content_cls,
            to_class_name=chunk_cls,
        )
    print('Chunk refs attached to content')
    print('')