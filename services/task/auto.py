import os
import sys
import argparse
import boto3

import weaviate as wv

# Tools
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

# Short Term Memory
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss

# Agent
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI


def main(task_list):
  auth_config = wv.auth.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])

  wv_client = wv.Client(
      url=os.environ['WEAVIATE_URL'],
      additional_headers={
          "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY']
      },
      auth_client_secret=auth_config
  )


  # Tool set up
  search = GoogleSerperAPIWrapper()

  # Tool definitions
  tools = [
      Tool(
          name = "search",
          func=search.run,
          description="Useful for questions about recent events or learning about things you are unsure about. You should ask targeted questions"
      ),
      WriteFileTool(),
      ReadFileTool(),
  ]


  # Define embedding model
  embeddings_model = OpenAIEmbeddings()
  embedding_size = 1536

  index = faiss.IndexFlatL2(embedding_size)

  vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


  # Define Agent
  agent = AutoGPT.from_llm_and_tools(
      ai_name='Oxpecker',
      ai_role='Assistant',
      tools=tools,
      llm=ChatOpenAI(temperature=0),
      memory=vectorstore.as_retriever()
  )
  # Set verbose to be true
  agent.chain.verbose = True


  # Run Agent
  agent.run(task_list)

  # Upload results to S3
  try:
    s3_client = boto3.client('s3')

    print(os.listdir('.'))

    for file_name in [f for f in os.listdir('.') if f.endswith('.txt')]:
      res = s3_client.upload_file(file_name, 'oxpecker.dev', f'auto/{file_name}')
  except:
    print('Skipping S3 upload in local mode')

  sys.exit()
  


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--task_list', type=str, nargs="*", default=["What's the weather tomorrow in Baltimore, MD?"])
      
  args, _ = parser.parse_known_args()
  print(args.task_list)

  main(args.task_list)