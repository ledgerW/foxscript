import unittest
from unittest_parametrize import param, parametrize, ParametrizedTestCase

import sys
sys.path.append('..')
from utils import workflow

class TestWorkflow(ParametrizedTestCase):
    @parametrize(
        ("body", "email"),
        [
            param({
                    'type': "LLM Prompt",
                    'init_text': "This is the prompt",
                    'as_list': True
                }, 
                "ledger.west@gmail.com",
                id="LLM_Prompt"
            ),
            param({
                    'type': "Send Output",
                    'destination': 'Project',
                    'as_workflow_doc': True,
                    'target_doc_input': "This is the document content",
                    'as_url_list': False
                }, 
                "ledger.west@gmail.com",
                id="Send_Output"
            ),
            param({
                    'type': "Combine"
                }, 
                "ledger.west@gmail.com",
                id="Combine"
            ),
        ],
    )
    def test_get_init(self, body, email):
        init_lengths = {
            'LLM Prompt': 2,
            'Send Output': 4,
            'Fetch Input': 1,
            'Combine': 0,
            'Analyze CSV': 1,
            'Web Research': 2,
            'Subtopics': 1,
            'Library Research': 5,
            'Workflow': 2
        }

        init = workflow.get_init(body, email)
        
        self.assertEqual(len(init), init_lengths[body['type']])


    @parametrize(
        ("body", "email"),
        [
            param({
                    'type': "Send Output",
                    'destination': 'Project',
                    'target_doc_input': "This is the document content",
                    'as_url_list': False
                }, 
                "ledger.west@gmail.com",
                id="Send_Output_missing_param"
            ),
        ],
    )
    def test_get_init_missing_param(self, body, email):
        with self.assertRaises(KeyError):
            init = workflow.get_init(body, email)

    #def test_isupper(self):
    #    self.assertTrue('FOO'.isupper())
    #    self.assertFalse('Foo'.isupper())


#if __name__ == '__main__':
#    unittest.main()