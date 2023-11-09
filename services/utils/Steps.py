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
import boto3
import pandas as pd
from datetime import datetime
#from youtubesearchpython import VideosSearch

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from utils.FoxLLM import FoxLLM, az_openai_kwargs, openai_kwargs
from utils.workflow_utils import get_top_n_search, get_context
from utils.weaviate_utils import wv_client
from utils.cloud_funcs import cloud_research
from utils.general import SQS


if os.getenv('IS_OFFLINE'):
    #boto3.setup_default_session(profile_name='ledger')
    lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
    LAMBDA_DATA_DIR = '.'
else:
    lambda_client = boto3.client('lambda')
    LAMBDA_DATA_DIR = '/tmp'

STAGE = os.getenv('STAGE')


class get_chain():
    def __init__(self, prompt=None, as_list=False):
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.as_list = as_list

        self.LLM = FoxLLM(az_openai_kwargs, openai_kwargs, model_name='gpt-4', temp=1.0)

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
            return res['text'].split('\n')
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
    def __init__(self, top_n=3):
        self.top_n = top_n
        self.input_word_cnt = 0
        self.output_word_cnt = 0
  

    def __call__(self, input):
        """
        Input: {'input': ["questions"]}

        Returns string
        """
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
        if os.getenv('IS_OFFLINE'):
            for url, query in zip(urls_to_scrape, queries):
                print('\nCloud Research call: {} - {}'.format(url, query))
                print('\n')

            return 'This is dummy web research from running locally'
        else:
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
    def __init__(self, class_name=None, k=3):
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.retriever = self.get_weaviate_retriever(class_name=class_name, k=k)

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


    def get_library_chunks(self, query):
        chunks = self.retriever.get_relevant_documents(query)

        return chunks


    def __call__(self, input):
        """
        Input: {'input': ["questions"]}

        Returns: string
        """
        questions = input['input']

        all_results = ''
        for question in questions[:5]:
            all_results = all_results + question + '\n'
            #chunks = self.get_library_chunks(question)
            #results = '\n'.join([c.page_content for c in chunks])
            
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

            all_results = all_results + results + '\n\n'
            time.sleep(3)

        # get input and output word count
        self.input_word_cnt = len(' '.join(questions).split(' '))
        self.output_word_cnt = len(all_results.split(' '))

        return all_results
  

class get_workflow():
    def __init__(self, workflow=None):
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        if workflow:
            self.workflow = workflow

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
                for input in input_vals:
                    payload = {
                        "body": {
                            'workflow_id': self.workflow.bubble_id,
                            'email': self.workflow.email,
                            'doc_id': '',
                            'run_id': '',
                            'input_vars': input_key,
                            'input_vals': input,
                            'sqs': sqs
                        }
                    }

                    print('\nWorkflow payload would be:')
                    print(payload)
                    print('\n')

                return 'Dummy OFFLINE Output'
            else:
                sqs = 'workflow{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
                queue = SQS(sqs)
                for input in input_vals:
                    payload = {
                        "body": {
                            'workflow_id': self.workflow.bubble_id,
                            'email': self.workflow.email,
                            'doc_id': '',
                            'run_id': '',
                            'input_vars': input_key,
                            'input_vals': input,
                            'sqs': sqs
                        }
                    }

                    _ = lambda_client.invoke(
                        FunctionName=f'foxscript-api-{STAGE}-workflow',
                        InvocationType='Event',
                        Payload=json.dumps(payload)
                    )
                    time.sleep(5)
            
                results = queue.collect(len(input_vals), max_wait=600)

                outputs = [result['output'] for result in results]
                self.input_word_cnt = sum([result['input_word_cnt'] for result in results])
                self.output_word_cnt = sum([result['output_word_cnt'] for result in results])
                return '\n\n'.join(outputs)
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
            'input_var2_name': "input_var2_val",
            ...
        }

        Returns: string
        """
        combined = '\n\n'.join([txt for txt in input.values()])

        # Get input and output word count
        self.input_word_cnt = len(combined.split(' '))
        self.output_word_cnt = len(combined.split(' '))

        return combined
        

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
    }
}