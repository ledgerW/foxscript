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

from pydantic import BaseModel, Field, create_model
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from utils.workflow_utils import get_top_n_search, get_context
from utils.weaviate_utils import wv_client
from utils.cloud_funcs import cloud_research
from utils.general import SQS


if os.getenv('IS_OFFLINE'):
    boto3.setup_default_session(profile_name='ledger')
    lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
    LAMBDA_DATA_DIR = '.'
else:
    lambda_client = boto3.client('lambda')
    LAMBDA_DATA_DIR = '/tmp'

STAGE = os.getenv('STAGE')


class get_chain():
    def __init__(self, prompt=None, as_list=False):
        self.as_list = as_list

        llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)

        self.input_vars = re.findall('{(.+?)}', prompt)

        _prompt = PromptTemplate(
            input_variables=self.input_vars,
            template=prompt
        )

        self.chain = LLMChain(llm=llm, prompt=_prompt, verbose=True)

    def __call__(self, input):
        """
        Input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val",
            ...
        }

        Returns: string
        """
        res = self.chain(input)

        if self.as_list:
            return res['text'].split('\n')
        else:
            return res['text']
    

class analyze_csv():
    def __init__(self, path=None):
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, verbose=True)

        df = pd.read_csv(path)
        
        file_name = path.split('/')[-1]
        df_shape = df.shape

        prefix = f"""You are working with a pandas dataframe in Python. The Python packages you have access to are pandas and numpy.
        The name of the dataframe is `df`.
        The df you are working with contains data from {file_name}
        This is the result of `print(df.shape)`: {df_shape}"""

        print(prefix)

        self.agent = create_pandas_dataframe_agent(
            llm,
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
            try:
                result = self.agent.run(question)
            except:
                result = "Problem answering this question."
            all_results = all_results + result + '\n\n'
            time.sleep(3)

        return all_results
    

class extract_from_text():
    def create_chain_parser_class(self, attributes):
        fields = {"__base__": BaseModel}

        for name, description in attributes.items():
            fields[name] = (List[str], Field(default=[], description=description))

        return create_model("ChainParser", **fields)
  
  
    def __init__(self, attributes=None):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1.0)

        self.attributes = attributes
        ChainParser = self.create_chain_parser_class(attributes)

        self.parser = PydanticOutputParser(pydantic_object=ChainParser)

        prompt = PromptTemplate(
            template="{format_instructions}\n\nGiven the below text, extract the information described in the output schema.\n{input}",
            input_variables=["input"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        self.chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


    def __call__(self, input):
        """
        Input: {'input': "input"}

        Returns: List[string]
        """
        input['input'] = "Original Text:\n" + input['input']
        res = self.chain(input)
        parsed_output = self.parser.parse(res['text']).dict()
        
        return parsed_output[list(self.attributes.keys())[0]]
  

class do_research():
    def __init__(self, top_n=3):
        self.top_n = top_n
  

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
        sqs = 'research{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
        queue = SQS(sqs)
        for url, query in zip(urls_to_scrape, queries):
            cloud_research(url, sqs, query)
            time.sleep(3)
        
        # wait for and collect scrape results from SQS
        research_context = queue.collect(len(urls_to_scrape), max_wait=600)
        return '\n'.join(research_context)
  

class get_library_retriever():
    def __init__(self, class_name=None, k=3):
        self.retriever = self.get_weaviate_retriever(class_name=class_name, k=k)


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
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1.0)

        questions = input['input']

        all_results = ''
        for question in questions[:5]:
            all_results = all_results + question + '\n'
            #chunks = self.get_library_chunks(question)
            #results = '\n'.join([c.page_content for c in chunks])
            results = get_context(question, llm, self.retriever, library=True)
            all_results = all_results + results + '\n\n'
            time.sleep(3)

        return all_results
  

class get_workflow():
    def __init__(self, workflow=None):
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
        if type(input_vals) == list:
            sqs = 'workflow{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
            queue = SQS(sqs)
            for input in input_vals:
                payload = {
                    "body": {
                    'workflow_id': self.workflow.bubble_id,
                    'email': self.workflow.email,
                    'doc_id': '',
                    'input_vars': input_key,
                    'input_vals': input,
                    'sqs': sqs
                    }
                }

                if os.getenv('IS_OFFLINE'):
                    print('\nWorkflow payload would be:')
                    print(payload)
                    print('\n')
                else:
                    _ = lambda_client.invoke(
                        FunctionName=f'foxscript-api-{STAGE}-workflow',
                        InvocationType='Event',
                        Payload=json.dumps(payload)
                    )
                    time.sleep(5)
          
            if os.getenv('IS_OFFLINE'):
                return 'Dummy OFFLINE Output'
            else:
                workflow_outputs = queue.collect(len(input_vals), max_wait=600)
                return '\n\n'.join(workflow_outputs)
        else:
            self.workflow.run_all(inputs)
            return self.workflow.steps[-1].output
        

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
    'Extract From Text': {
        'func': extract_from_text,
        'returns': 'list'
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
    }
}