import sys
sys.path.append('..')
sys.path.append('services')

import os
import json
import time
import requests

from utils.weaviate_utils import get_wv_class_name, delete_library
from utils.bubble import update_bubble_object, get_bubble_doc, delete_bubble_object
from utils.Steps import ACTIONS

if os.getenv('IS_OFFLINE'):
  LAMBDA_DATA_DIR = '.'
else:
  LAMBDA_DATA_DIR = '/tmp'

BUBBLE_API_KEY = os.getenv('BUBBLE_API_KEY')
BUBBLE_API_ROOT = os.getenv('BUBBLE_API_ROOT')


def get_init(body, email):
    if body['type'] == 'LLM Prompt':
        init = {
            'prompt': body['init_text'],
            'as_list': body['as_list']
        }

    if body['type'] == 'Send Output':
        init = {
            'destination': body['destination'],
            'as_workflow_doc': body['as_workflow_doc'],
            'target_doc_input': body['target_doc_input'],
            'as_url_list': False if 'as_url_list' not in body else body['as_url_list']
        }

    if body['type'] == 'Fetch Input':
        init = {
            'source': body['source']
        }

    if body['type'] == 'Combine':
        init = {}

    if body['type'] == 'Analyze CSV':
        doc_url = body['init_text']
        doc_file_name = doc_url.split('/')[-1]
        local_doc_path = f'{LAMBDA_DATA_DIR}/{doc_file_name}'
        get_bubble_doc(doc_url, local_doc_path)
        init = {'path': local_doc_path}

    if body['type'] == 'Web Research':
        init = {
            'top_n': int(body['init_number']),
            'web_qa': body['web_qa']
        }

    if body['type'] == 'Subtopics':
        init = {'top_n': int(body['init_number'])}

    if body['type'] == 'Library Research':
        class_name, account = get_wv_class_name(email, body['init_text'])
        init = {
            'class_name': class_name,
            'k': int(body['init_number']),
            'as_qa': body['as_qa'],
            'from_similar_docs': body['from_similar_docs'],
            'ignore_url': body['ignore_url']
        }

    if body['type'] == 'Workflow':
        init = {
            'workflow': get_workflow_from_bubble(body['init_text'], email=email),
            'in_parallel': body['in_parallel']
        }

    return init


def prep_input_vals(input_vars, input_vals, input):
    # prep for a Workflow
    if hasattr(input, 'steps'):
        input_type = input.steps[0].config['action']

        if input_type == 'LLM Prompt':
            input_vals = {var: source for var, source in zip(input_vars, input_vals)}  

        if input_type == 'Web Research':
            try:
                input_vals = {input_vars[0]: [x.split('\n') for x in input_vals]}
            except:
                input_vals = {input_vars[0]: input_vals[0]}

        if input_type == 'Subtopics':
            input_vals = {input_vars[0]: input_vals[0]}
        
        if input_type == 'Library Research':
            if len(input_vars) == 1:
                try:
                    input_vals = {input_vars[0]: [x.split('\n') for x in input_vals]}
                except:
                    input_vals = {input_vars[0]: input_vals[0]}
            else:
                try:
                    input_vals = {
                        input_vars[0]: [input_vals[0].split('\n')],
                        input_vars[1]: input_vals[1]
                    }
                except:
                    input_vals = {
                        input_vars[0]: input_vals[0],
                        input_vars[1]: input_vars[1]
                    }

        if input_type == 'Analyze CSV':
            try:
                input_vals = {input_vars[0]: [x.split('\n') for x in input_vals]}
            except:
                input_vals = {input_vars[0]: input_vals[0]}

        if input_type == 'Workflow':
            input_vals = {input_vars[0]: input_vals[0]}
    # prep for Step
    else:
        input_type = input.config['action']
        if input_type == 'LLM Prompt' or input_type == 'Combine':
            input_vals = {var: source for var, source in zip(input_vars, input_vals)}  

        if input_type == 'Web Research':
            input_vals = {'input': input_vals[0].split('\n')}

        if input_type == 'Subtopics':
            input_vals = {'input': input_vals[0]}
        
        if input_type == 'Library Research':
            if len(input_vars) == 1:
                try:
                    input_vals = {input_vars[0]: [x.split('\n') for x in input_vals]}
                except:
                    input_vals = {input_vars[0]: input_vals[0]}
            else:
                try:
                    input_vals = {
                        input_vars[0]: [input_vals[0].split('\n')],
                        input_vars[1]: input_vals[1]
                    }
                except:
                    input_vals = {
                        input_vars[0]: input_vals[0],
                        input_vars[1]: input_vars[1]
                    }

        if input_type == 'Analyze CSV':
            input_vals = {'input': input_vals[0].split('\n')}

        if input_type == 'Workflow':
            input_vals = {'input': input_vals[0].split('\n')}

    return input_vals


def step_config_from_bubble(bubble_step, email):
    print('\nbubble_step:')
    try:
        print(bubble_step)
    except:
        pass
    step_config = {
        "name": bubble_step['name'],
        "step": bubble_step['step_number'],
        "action": bubble_step['type'],
        "init": get_init(bubble_step, email),
        "inputs": {var: src for var, src in zip(bubble_step['input_vars'], bubble_step['input_vars_sources']) if var},
        "bubble_id": bubble_step['_id'],
        "output_type": "string" if bubble_step['type'] != 'Extract From Text' else 'list'
    }

    return step_config


def get_step_from_bubble(step_id, email=None, return_config=False):
    endpoint = BUBBLE_API_ROOT + '/step' + f'/{step_id}'
    res = requests.get(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'}
    )
    bubble_step = json.loads(res.content)['response']

    if return_config:
        return step_config_from_bubble(bubble_step, email)
    else:
        return Step(step_config_from_bubble(bubble_step, email))


def get_workflow_from_bubble(workflow_id, email=None, doc_id=None):
    # get workflow data from bubble db
    endpoint = BUBBLE_API_ROOT + '/workflow' + f'/{workflow_id}'
    res = requests.get(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'}
    )
    bubble_workflow = json.loads(res.content)['response']

    step_configs = [get_step_from_bubble(uid, email, return_config=True) for uid in bubble_workflow['steps']]

    workflow_config = {
        'name': bubble_workflow['name'],
        'steps': step_configs,
        'workflow_id': workflow_id,
        'email': email,
        'doc_id': doc_id
    }

    return Workflow().load_from_config(workflow_config)


class Workflow():
    def __init__(self, name=None):
        self.name = name
        self.steps = []
        self.output = {}
        self.user_inputs = {}
        self.email = None
        self.doc_id = None
        self.bubble_id = None
        self.workflow_library = None
        self.workflow_document = None
        self.input_word_cnt = 0
        self.output_word_cnt = 0

    def __repr__(self):
        step_repr = ["  Step {}. {}".format(i+1, s.name) for i, s in enumerate(self.steps)]
        return "\n".join([f'{self.name} Workflow'] + step_repr)

    def add_step(self, step):
        self.steps.append(step)

    def load_from_config(self, config):
        self.__init__(config['name'])

        try:
            self.bubble_id = config['workflow_id']
            self.email = config['email']
            self.doc_id = config['doc_id']
        except:
            pass
        
        for step_config in config['steps']:
            self.add_step(Step(step_config))

        return self

    def dump_config(self):
        return {
            "name": self.name,
            "steps": [s.config for s in self.steps]
        }
    
    def get_input_from_source(self, input_source, input_type):
        print(input_type)
        print(input_source)
        if "User Input" in input_source:
            input = self.user_inputs[input_source]

            if input_type == "Web Research":
                input = input[0]

            if input_type == "Library Research":
                input = input[0]

            if input_type == "Analyze CSV":
                input = input[0]

            return input
        else:
            step = [s for s in self.steps if s.name == input_source][0]
            return step.output

            
    def run_step(self, step_number, user_inputs={}):
        """
        input (str or list(str)): the input to the step (not in dictionary form) 
        """
        self.user_inputs = user_inputs
        
        # Get Step
        step = self.steps[step_number-1]
        print('\n')
        print('{} - {} - {}'.format(step_number, step.config['step'], step.name))

        step_input = {
            input_var: self.get_input_from_source(input_source, step.config['action']) for input_var, input_source in step.config['inputs'].items()
        }

        step.run_step(step_input)


    def run_all(self, user_inputs, bubble=False):
        """
        user_inputs (dict)
        """
        self.user_inputs = user_inputs

        print('user_inputs')
        print(self.user_inputs)
        
        for step in self.steps:
            print('{} - {}'.format(step.config['step'], step.name))
            is_workflow_step = step.config['action'] == 'Workflow'

            if is_workflow_step:
                print('doing Workflow Step')
                input_var, input_source = list(step.config['inputs'].items())[0]
                step_workflow_input_var = list(step.func.workflow.steps[0].config['inputs'].values())[0]
                step_workflow_input_val = self.get_input_from_source(input_source, step.config['action'])
                step_input = prep_input_vals([step_workflow_input_var], [step_workflow_input_val], step.func.workflow)
            else:
                print('doing Normal Step')
                print('input_var and source: {}'.format(step.config['inputs'].items()))
                step_input = {
                    input_var: self.get_input_from_source(input_source, step.config['action']) for input_var, input_source in step.config['inputs'].items()
                }
                print('step_input: {}'.format(step_input))


            # Update Step status in Bubble
            if bubble:
                bubble_body = {}
                bubble_body['is_running'] = True
                bubble_body['is_waiting'] = False
                bubble_body['unseen_output'] = False
                _ = update_bubble_object('step', step.bubble_id, bubble_body)


            # Attach Workflow Items to Step and Step func
            step.doc_id = self.doc_id
            step.workflow_name = self.name
            step.email = self.email
            step.func.doc_id = self.doc_id
            step.func.workflow_name = self.name
            step.func.email = self.email


            if self.workflow_library:
                step.workflow_library = self.workflow_library
                step.func.workflow_library = self.workflow_library

            if self.workflow_document:
                step.workflow_document = self.workflow_document
                step.func.workflow_document = self.workflow_document

            # Run the Step
            step.run_step(step_input)
            time.sleep(1)
            try:
                print(step.output[:1000])
            except:
                pass

            # Get Workflow Library, if there is one
            if step.workflow_library:
                self.workflow_library = step.workflow_library

            # Get Workflow Document, if there is one
            if step.workflow_document:
                self.workflow_document = step.workflow_document

            # Update workflow running total word usage
            self.input_word_cnt = self.input_word_cnt + step.input_word_cnt
            self.output_word_cnt = self.output_word_cnt + step.output_word_cnt

            # Write each step output back to Bubble
            if bubble:
                if type(step.output) == list:
                   output = '\n'.join(step.output)
                else:
                   output = step.output

                bubble_body = {}
                bubble_body['output'] = output
                bubble_body['input_word_cnt'] = step.input_word_cnt
                bubble_body['output_word_cnt'] = step.output_word_cnt
                bubble_body['is_running'] = False
                bubble_body['is_waiting'] = False
                bubble_body['unseen_output'] = True
                _ = update_bubble_object('step', step.bubble_id, bubble_body)

        # Finished running all steps
        if self.workflow_library:
            delete_library(self.workflow_library)
            print(f"Removed {self.workflow_library} from Weaviate")

        if self.workflow_document:
            delete_bubble_object('document', self.workflow_document)
            print(f"Removed Workflow Doc {self.workflow_document} from Bubble")
               


class Step():
    def __init__(self, config):
        self.name = config['name']
        self.config = config
        self.func = ACTIONS[config['action']]['func'](**config['init'])
        self.func.step_name = self.name
        self.doc_id = None
        self.output_type = config['output_type']
        self.bubble_id = config['bubble_id']
        self.output = None
        self.input_word_cnt = 0
        self.output_word_cnt = 0
        self.email = None
        self.workflow_library = None
        self.workflow_document = None

    def __repr__(self):
        return f'Step - {self.name}'

    def run_step(self, inputs):
        # Run it
        self.output = self.func(inputs)

        try:
            self.workflow_library = self.func.workflow_library
            print(f'Workflow has Library: {self.workflow_library}')
        except:
            print('No Workflow Library')

        try:
            self.workflow_document = self.func.workflow_document
            print(f'Workflow has Document: {self.workflow_document}')
        except:
            print('No Workflow Document')

        self.input_word_cnt = self.func.input_word_cnt
        self.output_word_cnt = self.func.output_word_cnt