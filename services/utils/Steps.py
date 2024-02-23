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
from langchain_community.embeddings import OpenAIEmbeddings

from utils.FoxLLM import FoxLLM, az_openai_kwargs, openai_kwargs
from utils.workflow_utils import (
    get_top_n_search,
    get_context,
    get_topic_clusters,
    get_cluster_results,
    get_cluster_results_by_source,
    get_keyword_batches,
    process_new_keyword
)
from utils.weaviate_utils import wv_client, get_wv_class_name, create_library
from utils.cloud_funcs import cloud_research, cloud_scrape
from utils.general import SQS
from utils.bubble import (
    create_bubble_object,
    update_bubble_object,
    get_bubble_object,
    upload_bubble_file,
    get_bubble_doc
)
from utils.google import convert_text, get_creds, create_google_doc, create_google_sheet, upload_to_google_drive
from utils.ghost import build_body, call_ghost, get_article_img


if os.getenv('IS_OFFLINE'):
    lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
    LAMBDA_DATA_DIR = '.'
else:
    lambda_client = boto3.client('lambda')
    LAMBDA_DATA_DIR = '/tmp'

STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')


"""
ALL STEP INPUT SHOULD BE DICT WITH 'input' AS STRING
OTHER KEY/VALUE PAIRS ALLOWED
"""


class get_chain():
    def __init__(self, prompt=None, as_list=False, split_on=None):
        self.split_on = split_on
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.as_list = as_list

        self.LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-4', temp=0.1)

        self.input_vars = re.findall('(?<!\{)\{([^{}]+)\}(?!\})', prompt)

        _prompt = PromptTemplate(
            input_variables=self.input_vars,
            template=prompt
        )

        self.chain = LLMChain(llm=self.LLM.llm, prompt=_prompt, verbose=True)


    def prep_input(self, input):
        '''
        input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val",
            ...
        }
        '''
        return input


    def __call__(self, input, TEST_MODE=False):
        """
        TESTING: Expect Prompt

        Input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val",
            ...
        }

        Returns: string
        """
        input = self.prep_input(input)
        
        if TEST_MODE:
            return input

        res = None
        try:
            res = self.chain(input)
        except:
            for llm in self.LLM.fallbacks:
                print(f'fallback to {llm}')
                self.chain.llm = llm
                try:
                    #res = self.chain(input)
                    res = self.chain.invoke(input)
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

        return res['text']
    

class analyze_csv():
    def __init__(self, path=None, split_on=None):
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.split_on = split_on
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


    def prep_input(self, input):
        if self.split_on:
            input['input'] = input['input'].split(self.split_on)
            input['input'] = [i for i in input['input'] if i]
        else:
            input['input'] = [input['input']]

        return input
    

    def __call__(self, input, TEST_MODE=False):
        """
        TESTING: Expect List of Prompts

        Input: {'input': [["questions"]]}

        Returns: string
        """
        input = self.prep_input(input)
        
        if TEST_MODE:
            return input

        questions = input['input']
        if type(questions) == str:
            questions = [questions]
    
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
    def __init__(self, top_n=3, web_qa=True, split_on=None):
        self.split_on = split_on
        self.top_n = top_n
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.web_qa = web_qa


    def prep_input(self, input):
        '''
        input: {'input' (str): "questions OR url"}
        '''
        if self.split_on:
            input['input'] = input['input'].split(self.split_on)
            input['input'] = [i for i in input['input'] if i]
        else:
            input['input'] = [input['input']]

        return input
  

    def __call__(self, input, TEST_MODE=False):
        """
        TESTING: Expect List of Prompts or URL

        Input: {'input': [["questions"] OR "URL"]}

        Returns string
        """
        input = self.prep_input(input)

        if TEST_MODE:
            return input
        
        if not self.web_qa:
            url = input['input'][0]

            res = cloud_scrape(url, sqs=None, invocation_type='RequestResponse', chunk_overlap=0, return_raw=True)
            res_body = json.loads(res['Payload'].read().decode("utf-8"))
            content = json.loads(res_body['body'])['chunks']

            self.input_word_cnt = 1
            self.output_word_cnt = len(content.split(' '))
            
            return content
        else:
            questions = input['input']
            if type(questions) == str:
                questions = [questions]

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
                for url in top_n_search_results['links']:
                    #url = _url['link']
                    urls_to_scrape.append(url)
                    queries.append(query)

            # Scrape and Research all URLs concurrently
            sqs = 'research{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
            queue = SQS(sqs)
            for url, query in zip(urls_to_scrape, queries):
                cloud_research(url, sqs, query)
                time.sleep(0.5)
        
            # wait for and collect scrape results from SQS
            results = queue.collect(len(urls_to_scrape), max_wait=600)

            outputs = [result['output'] for result in results]
            self.input_word_cnt = sum([result['input_word_cnt'] for result in results])
            self.output_word_cnt = sum([result['output_word_cnt'] for result in results])
            
            return '\n'.join(outputs)
  

class get_library_retriever():
    def __init__(self, class_name=None, k=3, as_qa=True, from_similar_docs=False, ignore_url=False, split_on=None):
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.split_on = split_on
        self.class_name = class_name
        self.k = k
        self.as_qa = as_qa
        self.from_similar_docs = from_similar_docs
        self.ignore_url = ignore_url
        self.retriever = self.get_weaviate_retriever(class_name=class_name, k=k)
        self.workflow_library = None

        self.LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-4', temp=0.1)


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
    

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def get_similar_content(self, nearVector, where_filter=None):
        result = wv_client.query\
            .get(f"{self.class_name}Content", ['title', 'source', 'url'])\
            .with_additional(["distance", 'id'])\
            .with_where(where_filter)\
            .with_near_vector(nearVector)\
            .with_limit(self.k)\
            .do()
        
        return result
    

    def prep_input(self, input):
        '''
        input: {'input': str}
        OR
        input: {
            'input': str,
            'URL To Ignore': 'https://urltoignore.com'
        }
        OR
        input: {
            'input': str,
            'Library From Input': 'Libraryname'
        }
        '''
        if self.split_on:
            input['input'] = input['input'].split(self.split_on)
            input['input'] = [i for i in input['input'] if i]
        else:
            input['input'] = [input['input']]

        return input


    def __call__(self, input, TEST_MODE=False):
        """
        TESTING: Expect List of Prompts

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
        input = self.prep_input(input)
        
        if TEST_MODE:
            return input

        print('Library Step Input')
        print(input)
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
        if type(questions) == str:
            questions = [questions]

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
                    print(f'Library Research with Similar Docs: {question}')
                    # get docs that are similar overall first
                    nearVector = {
                        "vector": OpenAIEmbeddings(model="text-embedding-3-large").embed_query(question)
                    }

                    if self.ignore_url:
                        url_to_ignore = input['URL To Ignore']
                        url_to_ignore = url_to_ignore[0] if type(url_to_ignore)==list else url_to_ignore
                    else:
                        url_to_ignore = 'DUMMY'

                    where_filter = {
                        "path": ["url"],
                        "operator": "NotEqual",
                        "valueText": url_to_ignore,
                    }

                    result = self.get_similar_content(nearVector, where_filter=where_filter)

                    articles = [{'source': res['source'], 'url': res['url'], 'id': res['_additional']['id']} for res in result['data']['Get'][f"{self.class_name}Content"]]
                    doc_urls = list(set([doc['url'] for doc in articles]))
                    print(f"doc urls: {doc_urls}")

                    # now get similar chunks only from overall similar docs
                    where_filter = {
                        "path": ["fromContent", f"{self.class_name}Content", "url"],
                        "operator": "ContainsAny",
                        "valueText": doc_urls
                    }

                    chunks = self.get_library_chunks(question, where_filter=where_filter)
                    results = '\n'.join([c.page_content for c in chunks])
                    print(f"Library Research Results: {results}")
                else:
                    chunks = self.get_library_chunks(question)
                    results = '\n'.join([c.page_content for c in chunks])

            all_results = all_results + results + '\n\n'
            time.sleep(0.5)

        # get input and output word count
        self.input_word_cnt = len(' '.join(questions).split(' '))
        self.output_word_cnt = len(all_results.split(' '))

        return all_results
    

class get_subtopics():
    def __init__(self, top_n=10, by_source=False, split_on=None):
        self.split_on = split_on
        self.top_n = top_n
        self.by_source = by_source
        self.input_word_cnt = 0
        self.output_word_cnt = 0

        self.LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-35-16k', temp=0.1)


    def prep_input(self, input):
        '''
        input: {'input': str}
        '''
        return input
  

    def __call__(self, input, TEST_MODE=False):
        """
        TESTING: Expect Keyword string

        Input: {'input': "topic"}

        Returns string
        """
        input = self.prep_input(input)

        if TEST_MODE:
            return input
        
        topic = input['input']

        topic_df = get_topic_clusters(topic, self.top_n)

        if self.by_source:
            subtopic_results, input_word_cnt, output_word_cnt = get_cluster_results_by_source(topic_df, self.LLM)
        else:
            subtopic_results, input_word_cnt, output_word_cnt = get_cluster_results(topic_df, self.LLM)

        self.input_word_cnt = input_word_cnt + len(topic.split(' '))
        self.output_word_cnt = output_word_cnt
        
        return subtopic_results
    

class cluster_keywords():
    def __init__(self, batch_size: int=100, thresh: float=0.8, split_on=None):
        self.split_on = split_on
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.batch_size = batch_size
        self.thresh = thresh


    def prep_input(self, input):
        '''
        input: {'input': str}
        '''
        return input
  

    def __call__(self, input, TEST_MODE=False):
        """
        TESTING: Expect Keyword string

        Input: {'input': "url/to/keyword_csv_name.csv"}

        Returns string
        """
        input = self.prep_input(input)

        if TEST_MODE:
            return input
        
        keyword_csv_url = input['input']
        keyword_csv_name = keyword_csv_url.split('/')[-1]
        local_keyword_csv_path = os.path.join(LAMBDA_DATA_DIR, keyword_csv_name)

        get_bubble_doc(keyword_csv_url, local_keyword_csv_path)
        
        keyword_batches = get_keyword_batches(local_keyword_csv_path, self.batch_size)
        print(len(keyword_batches))

        keyword_groups = []

        # run through each batch of keywords
        for i, keyword_batch in enumerate(keyword_batches):
            print(f"Batch {i+1}")
            print(f"Batch Size: {len(keyword_batch)}")
            sqs = 'googsearch{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
            queue = SQS(sqs)

            # do distributed google search for each keyword in batch
            counter = 0
            for q in keyword_batch:
                if q:
                    get_top_n_search(q, 10, sqs=sqs)
                    time.sleep(0.01)
                    counter += 1
                else:
                    pass

            # wait for and collect search results from SQS
            new_keywords = queue.collect(counter, max_wait=300)
            print(f"New Keyword Batch Size: {len(new_keywords)}")
            
            # group keywords form this batch
            for new_keyword in new_keywords:
                if len(new_keyword['links']) > 0:
                    keyword_groups = process_new_keyword(new_keyword, keyword_groups, thresh=self.thresh)

        keyword_groups_df = pd.DataFrame(keyword_groups)\
            .assign(group_size=lambda df: df.keywords.apply(len))

        print(f"Keyword Group Size: {keyword_groups_df.shape}")

        local_keyword_name = f"{keyword_csv_name.replace('.csv','')}_{int(self.thresh*100)}.csv"
        local_keyword_path = os.path.join(LAMBDA_DATA_DIR, local_keyword_name)
        keyword_groups_df.to_csv(local_keyword_path, index=False)
        bubble_url = upload_bubble_file(local_keyword_path)

        return bubble_url
  

class get_workflow():
    def __init__(self, workflow=None, in_parallel=True, split_on=None):
        self.split_on = split_on
        self.in_parallel = in_parallel
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        if workflow:
            self.workflow = workflow
            self.clean_workflow = workflow


    def prep_input(self, input):
        '''
        input: {'input': str}
        '''
        input_key = list(input.keys())[0]
        if self.split_on:
            print('SPLITTING')
            input[input_key] = input[input_key].split(self.split_on)
            input[input_key] = [i for i in input[input_key] if i]
        else:
            input[input_key] = [input[input_key]]

        return input
    

    def __call__(self, input, TEST_MODE=False):
        """
        Input: {
            'input_var': "input_var_val"
        }

        Returns: string
        """
        input = self.prep_input(input)
        print(f'input: {input}')

        if TEST_MODE:
            return input
        
        input_key = list(input.keys())[0]
        input_vals = list(input.values())[0]
        print('workflow step input vals:')
        print(input_vals)
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
                    time.sleep(0.5)
            
                results = queue.collect(len(input_vals), max_wait=900)

                outputs = [result['output'] for result in results]
                self.input_word_cnt = sum([result['input_word_cnt'] for result in results])
                self.output_word_cnt = sum([result['output_word_cnt'] for result in results])
                return '\n\n'.join(outputs)
            else:
                output = ''
                for order, input in enumerate(input_vals):
                    print(f'IN SEQUENCE WORKFLOW: {order}')
                    self.workflow.run_all([input_key], [input], bubble=False)
                    self.input_word_cnt = self.input_word_cnt + self.workflow.input_word_cnt
                    self.output_word_cnt = self.output_word_cnt + self.workflow.output_word_cnt
                    output = output + self.workflow.steps[-1].output + '\n\n'

                    # reset workflow
                    self.workflow = self.clean_workflow
                
                return output
        

class combine_output():
    def __init__(self, split_on=None):
        self.split_on = split_on
        self.input_word_cnt = 0
        self.output_word_cnt = 0


    def prep_input(self, input):
        '''
        input: {
            'input1': str,
            'input2': str
        }
        
        '''
        return input
    

    def __call__(self, input, TEST_MODE=False):
        """
        Input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val"
        }

        Returns: string
        """
        self.prep_input(input)

        if TEST_MODE:
            return input

        combined = '\n\n'.join([txt for txt in input.values()])

        # Get input and output word count
        self.input_word_cnt = len(combined.replace('\n\n', ' ').split(' '))
        self.output_word_cnt = len(combined.replace('\n\n', ' ').split(' '))

        return combined
    

class run_code():
    def __init__(self, py_code='', code_from_input=False, split_on=None):
        def exec_py_code(input, py_code):
            """py_code is python code to be exectued.
            It assumes an input value called 'input' and
            it returns a value called 'output' that must be
            defined in py_code.
            """
            py_code = f"""{py_code}"""
            exec_vals = {'input': input}
            
            try:
                exec(py_code, globals(), exec_vals)
                print(exec_vals)
                return exec_vals['output']
            except Exception as e:
                print(f'Error running py_code:\n{py_code}')
                print(e)
                return input
            
        
        self.split_on=split_on
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.code_from_input = code_from_input
        self.py_code = py_code
        self.py_func = exec_py_code


    def prep_input(self, input):
        '''
        Input: {'input': "text output from a previous Step"}
        OR
        Input: {
            'input': "text output from a previous Step",
            'py_code': "py code to execute from input source"
        }
        '''
        return input
            
        
    def __call__(self, input, TEST_MODE=False):
        """
        Input: {'input': "text output from a previous Step"}
        OR
        Input: {
            'input': "text output from a previous Step",
            'py_code': "py code to execute from input source"
        }

        Returns: string
        """
        input = self.prep_input(input)
        if TEST_MODE:
            return input
        
        if self.code_from_input:
            output = self.py_func(input['input'], input['py_code'])
        else:
            output = self.py_func(input['input'], self.py_code)

        # Get input and output word count
        self.input_word_cnt = len(input['input'].split(' '))
        self.output_word_cnt = len(output.replace('\n\n', ' ').split(' '))

        return output
    

class send_output():
    def __init__(
            self,
            destination=None,
            drive_folder='root',
            to_rtf=False,
            as_workflow_doc=False,
            empty_doc=False,
            target_doc_input=False,
            as_url_list=False,
            csv_doc=False,
            delimiter=',',
            with_post_image=True,
            publish_status='draft',
            template='custom-full-feature-image',
            split_on=None
        ):
        self.split_on = split_on
        self.destination = destination
        self.drive_folder = drive_folder
        self.to_rtf = to_rtf
        self.as_workflow_doc = as_workflow_doc
        self.empty_doc = empty_doc
        self.target_doc_input = target_doc_input
        self.as_url_list = as_url_list
        self.csv_doc = csv_doc
        self.delimiter = delimiter
        self.with_post_image = with_post_image
        self.publish_status = publish_status
        self.template = template
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.workflow_name = None
        self.step_name = None
        self.doc_id = None
        self.user_id = None
        self.email = None
        self.workflow_library = None
        self.workflow_document = None


    def prep_input(self, input):
        '''
        Input: {'input': "input_val"} or {'input': 'url\nurl\nurl'}
        or
        Input: {
            'input': "content",
            'Target Doc': '1234xIOS9erereoi'
        }
        Input: {
            'input': "content",
            'Title': 'doc title'
        }
        '''
        return input


    def __call__(self, input, TEST_MODE=False):
        """
        Input: {'input': "input_val"} or {'input': 'url\nurl\nurl'}
        or
        Input: {
            'input': "content",
            'Target Doc': '1234xIOS9erereoi'
        }
        Input: {
            'input': "content",
            'Title': 'doc title'
        }

        Returns: string
        """
        input = self.prep_input(input)

        if TEST_MODE:
            return input
        
        content = input['input']
        content = '\n'.join(content) if type(content)==list else content

        print(f'Destination: {self.destination}')
        print(f'As Workflow Doc: {self.as_workflow_doc}')
        print(f'Workflow Doc: {self.workflow_document}')

        # SEND TO FOXSCRIPT PROJECT
        if self.destination == 'Project':
            try:
                # get project id for output docs using dummy temp doc id provided in initial call
                res = get_bubble_object('document', self.doc_id)
                project_id = res.json()['response']['project']
            except:
                print('Passing. No Doc ID')
                pass
            
            # send result to Bubble Document
            if self.as_workflow_doc:
                if self.target_doc_input:
                    _ = update_bubble_object('document', input['Target Doc'], {'text': content})
                    new_doc_id = input['Target Doc']
                else:
                    try:
                        # Workflow Doc already exists and we're updating it
                        res = update_bubble_object('document', self.workflow_document, {'text': content})
                        resp = res.json()['response']
                        new_doc_id = self.workflow_document
                    except:
                        new_doc_name = 'tmp-workflow-doc'

                        body = {
                            'name': new_doc_name,
                            'text': '' if self.empty_doc else content,
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
                # Saving As Output
                new_doc_name = f"{self.step_name} - {input['Title']}"

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


        # SEND TO FOXSCRIPT WEAVIATE
        if self.destination == 'Workflow Library':
            lambda_client = boto3.client('lambda')

            # create new workflow library first time only
            # (will be destroyed at end of workflow)
            if not self.workflow_library:
                name = str(uuid.uuid4()).replace('-', '_')
                cls_name, account_name = get_wv_class_name(self.email, name)
                create_library(cls_name)
                self.workflow_library = cls_name
            
            if self.as_url_list:
                sqs = 'sendoutput{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
                queue = SQS(sqs)
                urls = [url for url in content.split('\n') if url]

                for url in urls:
                    print(f'Sending {url} to Workflow Library')
                    # load content into workflow library
                    out_body = {
                        'email': self.email,
                        'name': name,
                        'doc_url': url,
                        'sqs': sqs
                    }

                    print('RUNNING CLOUD')
                    _ = lambda_client.invoke(
                        FunctionName=f'foxscript-api-{STAGE}-upload_to_s3_cloud',
                        InvocationType='Event',
                        Payload=json.dumps({"body": out_body})
                    )

                results = queue.collect(len(urls), max_wait=600)
                return_value = self.workflow_library
            else:
                # load content into workflow library
                payload = {
                    "body": {
                        'bucket': BUCKET,
                        'cls_name': self.workflow_library,
                        'content': content
                    }
                }

                _ = lambda_client.invoke(
                    FunctionName=f'foxscript-data-{STAGE}-load_data',
                    InvocationType='RequestResponse',
                    Payload=json.dumps(payload)
                )

                return_value = self.workflow_library

        # GOOGLE DRIVE
        if self.destination == 'Google Drive':
            res = get_bubble_object('user', self.user_id)
            goog_token = res.json()['response']['DriveAccessToken']
            goog_refresh_token = res.json()['response']['DriveRefreshToken']
            
            creds = get_creds(goog_token, goog_refresh_token)

            title = input['Title'].replace(' ', '_')
            if self.csv_doc:
                sheet_id = create_google_sheet(
                    title,
                    content=content,
                    delimiter=self.delimiter,
                    folder_id=self.drive_folder,
                    creds=creds
                )
                drive_file_id = sheet_id
            else:
                if self.to_rtf:
                    rtf_content = convert_text(content, from_format='md', to_format='rtf')
                    file_id = upload_to_google_drive(
                        title,
                        'rtf',
                        content=rtf_content,
                        path=None,
                        folder_id=self.drive_folder,
                        creds=creds
                    )
                    drive_file_id = file_id
                else:
                    file_id = upload_to_google_drive(
                        title,
                        'md',
                        content=content,
                        path=None,
                        folder_id=self.drive_folder,
                        creds=creds
                    )

                    #doc_id = create_google_doc(
                    #    title,
                    #    content=content,
                    #    folder_id=self.drive_folder,
                    #    creds=creds
                    #)
                    
                    drive_file_id = file_id

            return_value = drive_file_id

        # CMS (Ghost)
        if self.destination.lower() == 'ghost':
            def get_content_tags(content: str, tags: list[str]) -> list[str]:
                prompt = f"""Please choose up to 3 tag options that best fit an article we want to publish.
                The tag is basically a category.

                Here is a sample of the Article:
                {content[:250]}

                
                And here are the Tag options:
                {tags}

                
                Please return only the chosen tag options (and nothing else). Return one tag option per line.
                If a tag is a location name, then put that tag at the top of the list.

                Tags:"""

                llm = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-4', temp=0.1)
                tags = llm.llm.invoke(prompt)
                
                return tags.content.split('\n')
            

            res = get_bubble_object('user', self.user_id)
            user_email = res.json()['response']['authentication']['email']['email']
            ghost_domain = res.json()['response']['ghost_domain']

            img_path = None
            if self.with_post_image:
                # get image url from google search and save to disk
                img_path = get_article_img(input['Title'] + 'high quality image', download=True)

                # upload local image file to ghost and get ghost url
                res = call_ghost(
                    user_id=self.user_id,
                    domain=ghost_domain,
                    endpoint_type='image',
                    img_path=img_path
                )
                print(res)
                img_path = res['images'][0]['url']

            # get tag options from Ghost Content API
            res = call_ghost(
                user_id=self.user_id,
                domain=ghost_domain,
                endpoint_type='tags'
            )
            tag_options = [tag['name'] for tag in res['tags']]
            tags = get_content_tags(input['Title'], tag_options)

            # get post body
            body = build_body(
                title=input['Title'],
                content=content,
                tags=tags,
                author_email=user_email,
                img_path=img_path,
                status=self.publish_status,
                template=self.template
            )

            # create post
            res = call_ghost(
                user_id=self.user_id,
                domain=ghost_domain,
                body=body,
                endpoint_type='post'
            )

            return_value = res['posts'][0]['id']

        # Get input and output word count
        self.input_word_cnt = len(content.replace('\n\n', ' ').split(' '))
        self.output_word_cnt = len(content.replace('\n\n', ' ').split(' '))

        return return_value
    

class fetch_input():
    def __init__(self, source=None, split_on=None):
        self.split_on = split_on
        self.source = source
        self.workflow_document = None
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.workflow_name = None
        self.step_name = None
        self.email = None


    def prep_input(self, input):
        '''
        input: {'input': str}
        '''

        return input


    def __call__(self, input, TEST_MODE=False):
        """
        Input: {'input': "input_val"}

        Returns: string
        """
        input = self.prep_input(input)

        if TEST_MODE:
            return input
        
        input = input['input']

        if self.source == 'Workflow Document':
            # get project id for output docs using dummy temp doc id provided in initial call
            try:
                res = get_bubble_object('document', self.workflow_document)
                content = res.json()['response']['text']
            except:
                content = ''
            
            return_value = content

        if self.source == 'Document From Input':
            # get project id for output docs using dummy temp doc id provided in initial call
            try:
                res = get_bubble_object('document', input)
                content = res.json()['response']['text']
            except:
                content = ''
            
            return_value = content

        # Get input and output word count
        self.input_word_cnt = len(content.replace('\n\n', ' ').split(' '))
        self.output_word_cnt = len(content.replace('\n\n', ' ').split(' '))

        return return_value
    
  

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
    'Run Code': {
        'func': run_code,
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
    },
    'Cluster Keywords': {
        'func': cluster_keywords,
        'returns': 'string'
    }
}