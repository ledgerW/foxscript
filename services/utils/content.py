import weaviate as wv
import pathlib
import json
import re
from urllib.parse import unquote
from datetime import datetime
from langchain_text_splitters import TokenTextSplitter
from PyPDF2 import PdfReader
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


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


def text_splitter(text, n, return_sentences=False, chunk_overlap=10):
    text_splitter = TokenTextSplitter(chunk_size=n, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)
    texts = [t.strip() for t in texts]

    return texts


def multi_page_text_splitter(pages, n):
    all_chunks = []
    page_nums = []
    chunk = ''
    for num, page in enumerate(pages):
        chunks = text_splitter(page, n)
        all_chunks = all_chunks + chunks
        page_nums = page_nums + [num+1 for _ in chunks]
    
    return all_chunks, page_nums


def handle_pdf(report_path, n, url=None, return_raw=False):
    reader = PdfReader(report_path)
    print(report_path.name)
    print('Found {} pages'.format(len(reader.pages)))

    pages = [p.extract_text() for p in reader.pages]
    chunks, page_nums = multi_page_text_splitter(pages, n)

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

    use_title = title if title != '' else report_path.name
    use_title = unquote(use_title)
    use_url = use_title if not url else url.split('/')[-1]
    use_url = unquote(use_url)
    meta = {
        'date': date,
        'url': use_url,
        'title': use_title,
        'author': author,
        'source': use_title
    }

    if return_raw:
        return pages, meta
    else:
        chunks, _ = multi_page_text_splitter(pages, n)
        for idx, chunk in enumerate(chunks):
            chunk_with_meta = f"""Source: {use_title}
            Author: {author}
            Date: {short_date}
            {chunk}
            """
            
            chunk_with_meta = re.sub(' +', ' ', chunk_with_meta)
            chunks[idx] = chunk_with_meta

        return chunks, page_nums, meta


def handle_json(report_path, n):
    with open(report_path, 'r') as file:
        report_json = json.load(file)
    
    chunks = text_splitter(report_json['content'], n)
    pages = [0 for _ in range(len(chunks))]

    title = report_json['title']
    author = report_json['author']
    date = report_json['date'].split('T')[0]
    url = report_json['url']

    for idx, chunk in enumerate(chunks):
        chunk_with_meta = f"""Title: {title}
        Author: {author}
        Date: {date}
        URL: {url}
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


def load_content(path, data_class=None, source=None, chunk_size=100, wv_client=None):
    report_path = pathlib.Path(path)

    content_cls = f'{data_class}Content'
    chunk_cls = f'{data_class}Chunk'

    if report_path.name.endswith('.pdf'):
        chunks, pages, meta = handle_pdf(report_path, chunk_size)
    elif report_path.name.endswith('.json'):
        chunks, pages, meta = handle_json(report_path, chunk_size)
        source = meta['source']

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