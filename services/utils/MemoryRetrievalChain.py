from langchain.base_language import BaseLanguageModel
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseRetriever, PromptValue
from langchain.memory.chat_memory import BaseChatMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain.callbacks.manager import CallbackManagerForChainRun

from utils.tools import get_intel_chunks, get_search_snippets

from typing import Dict, List, Any, Tuple, Optional

from pydantic import root_validator


class MemoryRetrievalChain(ConversationChain):
    prompt: PromptTemplate = None
    input_vars: List[str] = None
    retriever: BaseRetriever = None
    llm: BaseLanguageModel = None
    memory: BaseChatMemory = None
    max_token_limit: int = 2000
    input_key: str = 'input'
    retriever_input_key: str = 'context'
    search_input_key: str = 'search'
    intermediate_input_keys: List = []
    output_key: str = 'content'
    with_sources: bool = True
    sources: List = []
    pages: List = []

    # Defaults
    llm = llm or ChatOpenAI(temperature=0.5)

    intermediate_input_keys = [retriever_input_key, search_input_key]

    #memory = memory or ConversationTokenBufferMemory(
    #    input_key=input_key,
    #    llm=llm,
    #    max_token_limit=max_token_limit,
    #    memory_key='summary',
    #    human_prefix='User',
    #    ai_prefix='Oxpecker'
    #)


    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]
    

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        
        for inputs in input_list:
            print(inputs)
            # Get Library Context
            chunks = get_intel_chunks(self.retriever, inputs[self.input_key])
            context = '\n'.join([c.page_content for c in chunks])
            inputs['context'] = context

            # Get Google Search Context
            snippets, sources = get_search_snippets(inputs[self.input_key])
            inputs['search'] = '\n'.join(snippets)
            self.sources = self.sources + sources
            self.pages = self.pages + ['N/A' for _ in range(len(snippets))]

            # Get Wikipedia Search Context
            #snippets, sources = get_wiki_snippets(inputs[self.input_key])
            #inputs['wiki'] = '\n'.join(snippets)
            #self.sources = self.sources + sources
            #self.pages = self.pages + ['N/A' for _ in range(len(snippets))]

            # Get Prompt
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            
            _text = "Prompt after formatting:\n" + prompt.to_string()
            
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            
            prompts.append(prompt)
        
        return prompts, stop
    

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)

        if self.memory is not None:
            try:
                self.memory.save_context(inputs, outputs)
            except:
                self.memory.save_context({'input': inputs['input']}, outputs)
        
        if self.with_sources:
            return {**outputs, 'sources': self.sources, 'pages': self.pages}
        elif return_only_outputs:
            return {**outputs}
        else:
            return {**inputs, **outputs}
    

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        #memory_keys = values["memory"].memory_variables
        memory_keys = []
        input_key = values["input_key"]
        if input_key in memory_keys:
            raise ValueError(
                f"The input key {input_key} was also found in the memory keys "
                f"({memory_keys}) - please provide keys that don't overlap."
            )
        
        prompt_variables = values["prompt"].input_variables
        expected_keys = memory_keys + values["intermediate_input_keys"] + [input_key]
        if set(expected_keys) != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but got {memory_keys} as inputs from "
                f"memory, and {input_key} as the normal input key."
            )
        
        return values