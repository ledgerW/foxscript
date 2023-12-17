import sys
sys.path.append('..')

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import os
import json
import time
import re
import uuid
import boto3
import pandas as pd
from datetime import datetime
#from youtubesearchpython import VideosSearch

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.embeddings import OpenAIEmbeddings

from utils.FoxLLM import FoxLLM, az_openai_kwargs, openai_kwargs
from utils.workflow_utils import get_top_n_search, get_context, get_topic_clusters, get_cluster_results
from utils.weaviate_utils import wv_client, get_wv_class_name, create_library, to_json_doc
from utils.cloud_funcs import cloud_research, cloud_scrape
from utils.general import SQS
from utils.bubble import create_bubble_object, update_bubble_object, get_bubble_object


if os.getenv('IS_OFFLINE'):
    lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
    LAMBDA_DATA_DIR = '.'
else:
    lambda_client = boto3.client('lambda')
    LAMBDA_DATA_DIR = '/tmp'

STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')


class get_chain():
    def __init__(self, prompt=None, as_list=False):
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.as_list = as_list

        self.LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-4', temp=0.1)

        self.input_vars = re.findall('{(.+?)}', prompt)

        _prompt = PromptTemplate(
            input_variables=self.input_vars,
            template=prompt
        )

        self.chain = LLMChain(llm=self.LLM.llm, prompt=_prompt, verbose=True)

    def __call__(self, input):
        """
        Input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val",
            ...
        }

        Returns: string
        """
        res = None
        try:
            res = self.chain(input)
        except:
            for llm in self.LLM.fallbacks:
                print(f'fallback to {llm}')
                self.chain.llm = llm
                try:
                    res = self.chain(input)
                    if res:
                        break
                except:
                    continue

        if not res:
            res = {'text': 'Problem with Step.'}

        # Get input and output word count
        full_prompt = self.chain.prep_prompts([input])[0][0].text
        self.input_word_cnt = len(full_prompt.split(' '))
        self.output_word_cnt = len(res['text'].split(' '))

        if self.as_list:
            if "<SPLIT>" in res['text']:
                splitter = "<SPLIT>"
            else:
                splitter = "\n"
            return_items = res['text'].split(splitter)
            return [item for item in return_items if item != '']
        else:
            return res['text']
    

class analyze_csv():
    def __init__(self, path=None):
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-4', temp=0.1)

        df = pd.read_csv(path)
        
        file_name = path.split('/')[-1]
        df_shape = df.shape

        prefix = f"""You are working with a pandas dataframe in Python. The Python packages you have access to are pandas and numpy.
        The name of the dataframe is `df`.
        The df you are working with contains data from {file_name}
        This is the result of `print(df.shape)`: {df_shape}"""

        self.agent = create_pandas_dataframe_agent(
            self.LLM.llm,
            df,
            prefix=prefix,
            max_iterations=15,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

    def __call__(self, input):
        """
        Input: {'input': ["questions"]}

        Returns: string
        """
        questions = input['input']
    
        all_results = ''
        for question in questions[:5]:
            all_results = all_results + question + '\n'

            result = None
            try:
                result = self.agent.run(question)
            except:
                for llm in self.LLM.fallbacks:
                    self.agent.agent.llm = llm
                    try:
                        result = self.agent.run(question)
                        if result:
                            break
                    except:
                        continue
                
            if not result:
                result = "Problem answering this question."
                
            all_results = all_results + result + '\n\n'
            time.sleep(3)

        # get input and output word count
        self.input_word_cnt = len(' '.join(questions).split(' '))
        self.output_word_cnt = len(all_results.split(' '))

        return all_results
  

class do_research():
    def __init__(self, top_n=3, web_qa=True):
        self.top_n = top_n
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.web_qa = web_qa
  

    def __call__(self, input):
        """
        Input: {'input': ["questions OR URL"]}

        Returns string
        """
        if not self.web_qa:
            url = input['input'][0]

            res = cloud_scrape(url, sqs=None, invocation_type='RequestResponse', chunk_overlap=0)
            res_body = json.loads(res['Payload'].read().decode("utf-8"))
            content = json.loads(res_body['body'])['chunks'].replace('<SPLIT>', ' ')

            self.input_word_cnt = 1
            self.output_word_cnt = len(content.split(' '))
            
            return content
        else:
            questions = input['input']

            urls_to_scrape = []
            queries = []
            for _query in questions[:5]:
                query = _query.strip()
                print(query)

                # top n search context
                try:
                    top_n_search_results = get_top_n_search(query, self.top_n)
                except:
                    try:
                        time.sleep(3)
                        top_n_search_results = get_top_n_search(query, self.top_n)
                    except:
                        continue

                # Collect each url for this query
                for _url in top_n_search_results:
                    url = _url['link']
                    urls_to_scrape.append(url)
                    queries.append(query)

            # Scrape and Research all URLs concurrently
            sqs = 'research{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
            queue = SQS(sqs)
            for url, query in zip(urls_to_scrape, queries):
                cloud_research(url, sqs, query)
                time.sleep(3)
        
            # wait for and collect scrape results from SQS
            results = queue.collect(len(urls_to_scrape), max_wait=600)

            outputs = [result['output'] for result in results]
            self.input_word_cnt = sum([result['input_word_cnt'] for result in results])
            self.output_word_cnt = sum([result['output_word_cnt'] for result in results])
            
            return '\n'.join(outputs)
  

class get_library_retriever():
    def __init__(self, class_name=None, k=3, as_qa=True, from_similar_docs=False, ignore_url=False):
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.class_name = class_name
        self.k = k
        self.as_qa = as_qa
        self.from_similar_docs = from_similar_docs
        self.ignore_url = ignore_url
        self.retriever = self.get_weaviate_retriever(class_name=class_name, k=k)
        self.workflow_library = None

        self.LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-35-16k', temp=0.1)


    def get_weaviate_retriever(self, class_name=None, k=3):
        retriever = WeaviateHybridSearchRetriever(
            client=wv_client,
            index_name=f"{class_name}Chunk",
            text_key="chunk",
            k=k,
            attributes=['page', "fromContent {{... on {}Content {{ source url }}}}".format(class_name)],
            create_schema_if_missing=False
        )
        
        return retriever


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_library_chunks(self, query, where_filter=None):
        chunks = self.retriever.get_relevant_documents(query, where_filter=where_filter)

        return chunks


    def __call__(self, input):
        """
        Input: {'input': ["questions"]}
        or
        Input: {
            'input': ["questions"],
            'URL To Ignore': 'https://urltoignore.com'
        }
        or
        Input: {
            'input': ["questions"],
            'Library From Input': 'Libraryname'
        }

        Returns: string
        """
        if 'Workflow library' in self.class_name:
            print('Attempting to use Workflow Library')

            if self.workflow_library:
                self.retriever = self.get_weaviate_retriever(class_name=self.workflow_library, k=self.k)
                print(f'Using Worklfow Library: {self.workflow_library}')

        if 'Library from input' in self.class_name:
            print('Attempting to use Library From Input')

            self.retriever = self.get_weaviate_retriever(class_name=input['Library From Input'], k=self.k)
            print('Using Library From Input: {}'.format(input['Library From Input']))


        questions = input['input']

        all_results = ''
        for question in questions:
            all_results = all_results + "Query:\n" + question + '\n\n'
            
            if self.as_qa:
                results = None
                try:
                    results = get_context(question, self.LLM.llm, self.retriever, library=True)
                except:
                    for llm in self.LLM.fallbacks:
                        try:
                            results = get_context(question, llm, self.retriever, library=True)
                            if results:
                                break
                        except:
                            continue

                if not results:
                    results = 'Problem with Step.'
            else:
                if self.from_similar_docs:
                    # get docs that are similar overall first
                    nearVector = {
                        "vector": OpenAIEmbeddings().embed_query(question)
                    }

                    if self.ignore_url:
                        url_to_ignore = input['URL To Ignore']
                        url_to_ignore = url_to_ignore[0] if type(url_to_ignore)==list else url_to_ignore

                        where_filter = {
                            "path": ["url"],
                            "operator": "NotEqual",
                            "valueText": url_to_ignore,
                        }
                    else:
                        where_filter = {}

                    result = wv_client.query\
                        .get(f"{self.class_name}Content", ['title', 'source', 'url'])\
                        .with_additional(["distance", 'id'])\
                        .with_where(where_filter)\
                        .with_near_vector(nearVector)\
                        .with_limit(self.k)\
                        .do()

                    articles = [{'source': res['source'], 'url': res['url'], 'id': res['_additional']['id']} for res in result['data']['Get'][f"{self.class_name}Content"]]
                    doc_urls = list(set([doc['url'] for doc in articles]))

                    # now get similar chunks only from overall similar docs
                    where_filter = {
                        "path": ["fromContent", f"{self.class_name}Content", "url"],
                        "operator": "ContainsAny",
                        "valueText": doc_urls
                    }

                    chunks = self.get_library_chunks(question, where_filter=where_filter)
                    results = '\n'.join([c.page_content for c in chunks])
                else:
                    chunks = self.get_library_chunks(question)
                    results = '\n'.join([c.page_content for c in chunks])

            all_results = all_results + results + '\n\n'
            time.sleep(3)

        # get input and output word count
        self.input_word_cnt = len(' '.join(questions).split(' '))
        self.output_word_cnt = len(all_results.split(' '))

        return all_results
    

class get_subtopics():
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.input_word_cnt = 0
        self.output_word_cnt = 0

        self.LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-35-16k', temp=0.1)
  

    def __call__(self, input):
        """
        Input: {'input': "topic"}

        Returns string
        """
        topic = input['input']

        topic_df = get_topic_clusters(topic, self.top_n)

        subtopic_results, input_word_cnt, output_word_cnt = get_cluster_results(topic_df, self.LLM)

        self.input_word_cnt = input_word_cnt + len(topic.split(' '))
        self.output_word_cnt = output_word_cnt
        
        return subtopic_results
  

class get_workflow():
    def __init__(self, workflow=None, in_parallel=True):
        self.in_parallel = in_parallel
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        if workflow:
            self.workflow = workflow
            self.clean_workflow = workflow

    def __call__(self, inputs):
        """
        Input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val",
            ...
        }

        Returns: string
        """
        input_key = list(inputs.keys())[0]
        input_vals = list(inputs.values())[0]
        print('workflow step input vals:')
        print(input_vals)
        if type(input_vals) == list:
            if os.getenv('IS_OFFLINE'):
                sqs = 'workflow{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
                for order, input in enumerate(input_vals):
                    payload = {
                        "body": {
                            'workflow_id': self.workflow.bubble_id,
                            'email': self.workflow.email,
                            'doc_id': '',
                            'run_id': '',
                            'input_vars': input_key,
                            'input_vals': input,
                            'sqs': sqs,
                            'order': order
                        }
                    }

                    print('\nWorkflow payload would be:')
                    print(payload)
                    print('\n')

                return 'Dummy OFFLINE Output'
            else:
                if self.in_parallel:
                    sqs = 'workflow{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
                    queue = SQS(sqs)
                    for order, input in enumerate(input_vals):
                        payload = {
                            "body": {
                                'workflow_id': self.workflow.bubble_id,
                                'email': self.workflow.email,
                                'doc_id': '',
                                'run_id': '',
                                'input_vars': input_key,
                                'input_vals': input,
                                'sqs': sqs,
                                'order': order
                            }
                        }

                        _ = lambda_client.invoke(
                            FunctionName=f'foxscript-api-{STAGE}-workflow',
                            InvocationType='Event',
                            Payload=json.dumps(payload)
                        )
                        time.sleep(2)
                
                    results = queue.collect(len(input_vals), max_wait=780)

                    outputs = [result['output'] for result in results]
                    self.input_word_cnt = sum([result['input_word_cnt'] for result in results])
                    self.output_word_cnt = sum([result['output_word_cnt'] for result in results])
                    return '\n\n'.join(outputs)
                else:
                    output = ''
                    for order, input in enumerate(input_vals):
                        self.workflow.run_all({input_key: input}, bubble=False)
                        self.input_word_cnt = self.input_word_cnt + self.workflow.input_word_cnt
                        self.output_word_cnt = self.output_word_cnt + self.workflow.output_word_cnt
                        output = output + self.workflow.steps[-1].output + '\n\n'

                        # reset workflow
                        self.workflow = self.clean_workflow
                    
                    return output

        else:
            self.workflow.run_all(inputs, bubble=False)
            self.input_word_cnt = self.workflow.input_word_cnt
            self.output_word_cnt = self.workflow.output_word_cnt
            return self.workflow.steps[-1].output
        

class combine_output():
    def __init__(self):
        self.input_word_cnt = 0
        self.output_word_cnt = 0

    def __call__(self, input):
        """
        Input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val"
        }

        Returns: string
        """
        combined = '\n\n'.join([txt for txt in input.values()])

        # Get input and output word count
        self.input_word_cnt = len(combined.replace('\n\n', ' ').split(' '))
        self.output_word_cnt = len(combined.replace('\n\n', ' ').split(' '))

        return combined
    

class send_output():
    def __init__(self, destination=None, as_workflow_doc=False):
        self.destination = destination
        self.as_workflow_doc = as_workflow_doc
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.workflow_name = None
        self.step_name = None
        self.doc_id = None
        self.email = None
        self.workflow_library = None
        self.workflow_document = None


    def __call__(self, input):
        """
        Input: {'input': "input_val"}

        Returns: string
        """
        content = input['input']
        content = '\n'.join(content) if type(content)==list else content

        print(f'Destination: {self.destination}')
        print(f'As Workflow Doc: {self.as_workflow_doc}')
        print(f'Workflow Doc: {self.workflow_document}')

        if self.destination == 'Project':
            # get project id for output docs using dummy temp doc id provided in initial call
            res = get_bubble_object('document', self.doc_id)
            project_id = res.json()['response']['project']
            
            # send result to Bubble Document
            if self.as_workflow_doc:
                try:
                    res = update_bubble_object('document', self.workflow_document, {'text': content})
                    resp = res.json()['response']
                    new_doc_id = self.workflow_document
                except:
                    new_doc_name = 'tmp-workflow-doc'

                    body = {
                        'name': new_doc_name,
                        'text': content,
                        'user_email': self.email,
                        'project': project_id
                    }
                    res = create_bubble_object('document', body)
                    new_doc_id = res.json()['id'] 

                    self.workflow_document = new_doc_id

                    # add new doc to project
                    res = get_bubble_object('project', project_id)
                    try:
                        project_docs = res.json()['response']['documents']
                    except:
                        project_docs = []

                    _ = update_bubble_object('project', project_id, {'documents': project_docs+[new_doc_id]}) 
            else:
                new_doc_name = f"{self.step_name} - {content[:30]}"

                body = {
                    'name': new_doc_name,
                    'text': content,
                    'user_email': self.email,
                    'project': project_id
                }
                res = create_bubble_object('document', body)
                new_doc_id = res.json()['id']

                # add new doc to project
                res = get_bubble_object('project', project_id)
                try:
                    project_docs = res.json()['response']['documents']
                except:
                    project_docs = []

                _ = update_bubble_object('project', project_id, {'documents': project_docs+[new_doc_id]})

            return_value = new_doc_id


        if self.destination == 'Workflow Library':
            lambda_client = boto3.client('lambda')

            # create new workflow library (will be destroyed at end of workflow)
            name = str(uuid.uuid4()).replace('-', '_')
            cls_name, account_name = get_wv_class_name(self.email, name)
            create_library(cls_name)
            self.workflow_library = cls_name

            # load content into workflow library
            payload = {
                "body": {
                    'bucket': BUCKET,
                    'cls_name': cls_name,
                    'content': content

                }
            }

            _ = lambda_client.invoke(
                FunctionName=f'foxscript-data-{STAGE}-load_data',
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )

            return_value = cls_name

        # Get input and output word count
        self.input_word_cnt = len(content.replace('\n\n', ' ').split(' '))
        self.output_word_cnt = len(content.replace('\n\n', ' ').split(' '))

        return return_value
    

class fetch_input():
    def __init__(self, source=None):
        self.source = source
        self.workflow_document = None
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.workflow_name = None
        self.step_name = None
        self.doc_id = None
        self.email = None


    def __call__(self, input):
        """
        Input: {'input': "input_val"}

        Returns: string
        """
        input = input['input']

        if self.source == 'Workflow Document':
            # get project id for output docs using dummy temp doc id provided in initial call
            res = get_bubble_object('document', self.workflow_document)
            content = res.json()['response']['text']
            
            return_value = content

        if self.source == 'Document From Input':
            # get project id for output docs using dummy temp doc id provided in initial call
            res = get_bubble_object('document', input)
            content = res.json()['response']['text']
            
            return_value = content

        # Get input and output word count
        self.input_word_cnt = len(content.replace('\n\n', ' ').split(' '))
        self.output_word_cnt = len(content.replace('\n\n', ' ').split(' '))

        return return_value
        

#class get_yt_url():
#  def __init__(self, n=1):
#    self.n = n
#    self.input_vars = ['query']
  
#  def __call__(self, input):
#    """
#    Input: {'input': "query"}

#    Returns: string
#    """
#    query = input['input']

#    vid_search = VideosSearch(query, limit=self.n)

#    return vid_search.result()['result'][0]['link']
  

# Step Actions.
# A Step Action is a function that returns (func, [inputs_names])
ACTIONS = {
    'LLM Prompt': {
        'func': get_chain,
        'returns': 'string'
    },
    'Web Research': {
        'func': do_research,
        'returns': 'string'
    },
    'Library Research': {
        'func': get_library_retriever,
        'returns': 'string'
    },
    'Analyze CSV': {
        'func': analyze_csv,
        'returns': 'string'
    },
    'Workflow': {
        'func': get_workflow,
        'returns': 'string'
    },
    'Combine': {
        'func': combine_output,
        'returns': 'string'
    },
    'Subtopics': {
        'func': get_subtopics,
        'returns': 'string'
    },
    'Send Output': {
        'func': send_output,
        'returns': 'string'
    },
    'Fetch Input': {
        'func': fetch_input,
        'returns': 'string'
    }
}