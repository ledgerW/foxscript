import sys
sys.path.append('..')

from utils.workflow import Workflow


workflow_config = {'name': 'Child-Memo Highlights',
 'steps': [{'name': 'Fetch Document',
   'step': 1,
   'user_id': '0x1',
   'action': 'Web Research',
   'init': {'top_n': 1, 'web_qa': False, 'split_on': None},
   'inputs': {'input': 'User Input - input'},
   'bubble_id': '1705936185862x825661441418199000',
   'output_type': None,
   'output': """Name of Doc\n\nA whole bunch of text and article stuff.""",
   'expected_input': {'input': ['https://www.url.com']}},
  {'name': 'Document Highlights',
   'step': 2,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': "Document:\n{document}\n\n\nThe first line of the Document provided above is the title.\n\nYou are a credit analyst for Warbler Labs. You are preparing an internal investment memo for the Warbler Investment Committee. This memo should present all the important details, risks, and benefits of the loan opportunity. Naturally, specific numbers, dollar amounts, percentages, etc... are extremely important, and must presented accurately!\n\nPlease perform a breakdown analysis of the Document provided above. Return your breakdown analysis using the markdown table and bullet point format provided in the example below. The example below suggests the types of questions and information you should extract. If a specific piece of information isn't available or is unclear, please insert your best guess and simply note that you need to follow up on this piece of information.\n\nAnd be sure to cite the title of the Document at the end of your responses inside parentheses.\n\nExample:\n| Category | Term |\n| --- | --- |\n| Warbler Participation | [amount we're investing (out of total size of loan)] |\n| Interest | [interest rate(s)] |\n| Maturity | [maturity date(s)] |\n| Interest Freq. | [frequency] |\n\n### **Highlights**\n- What are the basic terms of the loan?\n- Who is the borrower?\n- Who is the lender?\n- Summarize the basic operations of the borrower - use specific numbers.\n- Is there any recent borrower activity worth highlighting, either good or bad?\n- Describe the corporate and legal relationship of the entities involved. Sometimes the borrower or lender have multiple legal entities involved in the loan. This is important to understand well.\n- What is the collateral of the loan? Please Describe it.\n- What are the financial characteristics of the collateral?\n- Have the borrower or lender engaged in transactions in the past? If so, describe those.\n- What is the operational track record of borrower?\n- What is the operational track record of the lender?\n- What are the primary risks of this transaction?\n- What are the less likely secondary risks of this transaction?\n- Highlight any other relevant information found in this document that wasn't addressed in the above questions.\n\n\nHighlights:",
    'as_list': False, 'split_on': None},
   'inputs': {'document': 'Fetch Document'},
   'bubble_id': '1705936724431x749227599029338100',
   'output_type': None,
   'output': """The Doc Highlights Output text""",
   'expected_input': {'document': """Name of Doc\n\nA whole bunch of text and article stuff."""}}]}


workflow = Workflow().load_from_config(workflow_config, TEST_MODE=True)