try:
  from langchain.utilities import WikipediaAPIWrapper
except:
  pass

from langchain.utilities import GoogleSerperAPIWrapper


def get_intel_chunks(retriever, query):
  chunks = retriever.get_relevant_documents(query)
  #sources = [c.metadata['fromReport'][0]['source'] for c in chunks]
  #pages = [c.metadata['page'] for c in chunks]

  return chunks#, sources, pages


def get_search_snippets(query):
    search = GoogleSerperAPIWrapper()
    search_result = search.results(query)
    
    snippets = []
    sources = []
    if 'knowledgeGraph' in search_result:
        snippets.append(search_result['knowledgeGraph']['description'])
        sources.append(search_result['knowledgeGraph']['descriptionLink'])
    if 'answerBox' in search_result:
        snippets.append(search_result['answerBox']['snippet'])
        sources.append(search_result['answerBox']['link'])
    if 'organic' in search_result:
        for organic in search_result['organic'][:3]:
            snippets.append(organic['snippet'])
            sources.append(organic['link'])
    if 'peopleAlsoAsk' in search_result:
        for also_ask in search_result['peopleAlsoAsk'][:5]:
            snippets.append(also_ask['snippet'])
            sources.append(also_ask['link'])

    return snippets, sources


def get_wiki_snippets(query):
  wiki = WikipediaAPIWrapper(top_k_results=3)
  wiki_pages = wiki.wiki_client.search(query)

  snippets = []
  sources = []
  for page in wiki_pages:
    sum = wiki.wiki_client.summary(page)
    
    if sum != '':
      sum = ' '.join(sum.split(' ')[:75])
      snippets.append(sum)
      sources.append(f'Wikipedia - {page}')


    if len(snippets) == 3:
      break

  return snippets, sources