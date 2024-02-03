import sys
sys.path.append('..')

from utils.workflow import Workflow
from .child_memo_highlights import workflow as child_memo_highlights_wf


workflow_config = {'name': 'Memo Highlights',
 'steps': [{'name': 'Get List of URLs',
   'step': 1,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'List of URLs:\n{urls}\n\nReturn the the above list of urls, one per line.  Don\'t include anything else, like """ or \'\'\'.\n\nURLs:',
    'as_list': False, 'split_on': None},
   'inputs': {'urls': 'User Input - urls'},
   'bubble_id': '1705975527169x577871191234052100',
   'output_type': None,
   'output': 'https://www.url.com\nhttps://www.url2.com',
   'expected_input': {'urls': 'https://www.url.com\nhttps://www.url2.com'}},
  {'name': 'Memo Highlights',
   'step': 2,
   'user_id': '0x1',
   'action': 'Workflow',
   'init': {'workflow': child_memo_highlights_wf,
    'in_parallel': True, 'split_on': '\n'},
   'inputs': {'input': 'Get List of URLs'},
   'bubble_id': '1705975527169x445888616273018900',
   'output_type': None,
   'output': """The Doc Highlights Output text""",
   'expected_input': {'User Input - input': ['https://www.url.com', 'https://www.url2.com']}},
  {'name': 'Final Highlights',
   'step': 3,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': "Draft Highlights:\n{highlights}\n\n\nYou are a credit analyst for Warbler Labs. You are preparing an internal investment memo for the Warbler Investment Committee. This memo should present all the important details, risks, and benefits of the loan opportunity. Naturally, specific numbers, dollar amounts, percentages, etc... are extremely important, and must be presented accurately!\n\nAbove, you've been provided with multiple Draft Highlights for our internal memo. In this case, each of the Draft Highlight provided above was completed using a different primary deal document, and your job is to aggregate each of the Draft Highlights into one final Highlights that contains all the information from each of the Draft Highlights. Note, DO NOT simply append all the responses into one final highlights. You want to aggregate and edit all of the responses into a single best response that references specific numbers and figures. Include specific numbers and figures as much as possible. If a section or line is blank or doesn't have a response in a particular Highlights that just means the document used for that Highlights didn't have the information, and you should use the information that another Highlights has for that section or line. It doesn't mean there is uncertainty in the response.\n\nOtherwise, do not alter the Highlights format.\n\nYou'll note each response cites the document it used in parentheses at the end of the response.\nBe sure to include the cited document names for the responses you ended up using in the final version. Put them in parentheses at the end of the response.\n\n\nFinal Highlights (don't actually write this):",
    'as_list': False, 'split_on': None},
   'inputs': {'highlights': 'Memo Highlights'},
   'bubble_id': '1705975527169x274900185689358340',
   'output_type': None,
   'output': """Final aggregated Highlights Output text""",
   'expected_input': {'highlights': """The Doc Highlights Output text"""}}]}


workflow = Workflow().load_from_config(workflow_config, TEST_MODE=True)