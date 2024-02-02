import unittest
from unittest_parametrize import parametrize, ParametrizedTestCase

import sys
sys.path.append('..')

from fixtures import workflow_params


class TestWorkflowIO(ParametrizedTestCase):
    @parametrize(
        workflow_params.io_fixture['params'],
        workflow_params.io_fixture['values']
    )
    def test_workflow_input(self, workflow, input_var: [str], input_val: [str]):
        """
        Testing the post prep_input input to a Step's function.
        """
        self.maxDiff = None

        workflow.run_all(input_var, input_val, bubble=False)
        
        for num, step in enumerate(workflow.steps):
            self.assertEqual(step.TEST_STEP_INPUT, step.TEST_EXPECTED_INPUT, f"{num+1}-{step.name}")