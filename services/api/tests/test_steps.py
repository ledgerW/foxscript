import unittest
from unittest_parametrize import parametrize, ParametrizedTestCase

import sys
sys.path.append('..')

from utils.Steps import (
    get_chain,
    get_library_retriever,
    do_research
)

from fixtures import (
    step_chain,
    step_library,
    step_web
)


class TestStepChain(ParametrizedTestCase):
    @parametrize(
        step_chain.input_fixture['params'],
        step_chain.input_fixture['values']
    )
    @unittest.skip('Local Testing')
    def test_get_chain_input(self, init, input, check_input, check_prompt):
        chain = get_chain(**init)
        _input = chain.prep_input(input)
        _prompt = chain.chain.prompt.format_prompt(**_input)
        
        self.assertEqual(_input, check_input)
        self.assertEqual(_prompt.text, check_prompt)


    @parametrize(
        step_chain.call_fixture['params'],
        step_chain.call_fixture['values']
    )
    @unittest.skip('Local Testing')
    def test_get_chain_call(self, init, input):
        chain = get_chain(**init)
        try:
            result = chain(input)
        except:
            result = None
        
        self.assertIsNotNone(result)


class TestStepLibrary(ParametrizedTestCase):
    @parametrize(
        step_library.input_fixture['params'],
        step_library.input_fixture['values']
    )
    @unittest.skip('Local Testing')
    def test_library_input(self, init, input, check_input):
        library = get_library_retriever(**init)
        _input = library.prep_input(input)
        
        self.assertEqual(_input, check_input)


class TestStepWeb(ParametrizedTestCase):
    @parametrize(
        step_web.input_fixture['params'],
        step_web.input_fixture['values']
    )
    @unittest.skip('Local Testing')
    def test_web_input(self, init, input, check_input):
        web = do_research(**init)
        _input = web.prep_input(input)
        
        self.assertEqual(_input, check_input)

'''
class TestStepWorkflow(ParametrizedTestCase):
    @parametrize(
        workflow_params.input_fixture['params'],
        workflow_params.input_fixture['values']
    )
    @unittest.skip('Local Testing')
    def test_workflow_input(self, init, input, check_input):
        workflow = get_workflow(**init)
        _input = workflow.prep_input(input)
        
        self.assertEqual(_input, check_input)
'''