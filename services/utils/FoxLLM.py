import sys
sys.path.append('..')

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI


AZ_GPT_35 = 'gpt-35-turbo'
AZ_GPT_35_16K = 'gpt-35-turbo-16k'
AZ_GPT_4 = 'gpt-4'
AZ_GPT_4_32K = 'gpt-4-32k'
AZ_EMBEDDING = 'text-embedding-ada-002'

GPT_35 = 'gpt-3.5-turbo'
GPT_35_16K = 'gpt-3.5-turbo-16k'
GPT_4 = 'gpt-4'
EMBEDDING = 'text-embedding-ada-002'

az_openai_kwargs = {
    'openai_api_base': os.getenv('AZURE_OPENAI_BASE'),
    'openai_api_version': "2023-05-15",
    'openai_api_key': os.getenv('AZURE_OPENAI_API_KEY'),
    'openai_api_type': "azure"
}

openai_kwargs = {
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'openai_organization': os.getenv('OPENAI_ORGANIZATION')
}


class FoxLLM():
    def __init__(self, az_openai_kwargs, openai_kwargs, model_name=None, temp=1.0):
        self.model_name = model_name

        self.az_models = {
            'gpt-35': AzureChatOpenAI(**az_openai_kwargs, deployment_name=AZ_GPT_35, model_name=AZ_GPT_35, temperature=temp, verbose=True),
            'gpt-35-16k': AzureChatOpenAI(**az_openai_kwargs, deployment_name=AZ_GPT_35_16K, model_name=AZ_GPT_35_16K, temperature=temp, verbose=True),
            'gpt-4': AzureChatOpenAI(**az_openai_kwargs, deployment_name=AZ_GPT_4, model_name=AZ_GPT_4, temperature=temp, verbose=True),
            'gpt-4-32k': AzureChatOpenAI(**az_openai_kwargs, deployment_name=AZ_GPT_4_32K, model_name=AZ_GPT_4_32K, temperature=temp, verbose=True)
        }

        self.openai_models = {
            'gpt-35': ChatOpenAI(**openai_kwargs, model_name=GPT_35, temperature=temp, verbose=True),
            'gpt-35-16k': ChatOpenAI(**openai_kwargs, model_name=GPT_35_16K, temperature=temp, verbose=True),
            'gpt-4': ChatOpenAI(**openai_kwargs, model_name=GPT_4, temperature=temp, verbose=True)
        }

        self.model_fallbacks = {
            'gpt-4': [
                self.az_models['gpt-35-16k'],
                self.az_models['gpt-35'],
                self.openai_models['gpt-4'],
                self.az_models['gpt-4-32k']
            ],
            'gpt-35-16k': [
                self.openai_models['gpt-35'],
                self.az_models['gpt-35'],
                self.openai_models['gpt-35-16k']
            ]
        }

        self.llm = self.az_models[model_name]
        self.fallbacks = list(reversed(self.model_fallbacks[model_name]))

    def fallback(self):
        self.llm = self.model_fallbacks[self.model_name].pop()