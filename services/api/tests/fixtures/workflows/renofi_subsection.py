import sys
sys.path.append('..')

from utils.workflow import Workflow


workflow_config = {'name': 'V2 RenoFi Subsection',
 'steps': [{'name': 'Get Outline Subsection Only',
   'step': 1,
   'action': 'LLM Prompt',
   'init': {'prompt': 'Provided:\n{outline sub section}\n\nThe information provided above uses the following format:\nLibrary Name: Libraryname\nTopic: Topic (in format of all lowercase with hyphens and no special characters)\nOutline Subsection:\n### Subsection\n#### Subsection Topic\n\nPlease extract the Topic and Outline Subsection, and return ONLY the Topic and Outline Subsection exactly as is! Topic should be formatted as all lowercase with hyphens and no special characters.\n\nIgnore the Library Name.\n\nTopic and Outline Subsection:',
    'as_list': False, 'split_on': None},
   'inputs': {'outline sub section': 'User Input - outline sub section'},
   'bubble_id': '1705855613936x605555398488883200',
   'output_type': None,
   'output': """Topic: accessory-dwelling-unit-adu
Outline Subsection:
### Community Engagement
#### Volunteering for Housing Projects""",
   'expected_input': {'outline sub section': 'Library Name: LedgerwestgmailE49d4d23_4279_4fcf_ba76_727a53d9df1e\nTopic: accessory-dwelling-unit-adu\nOutline Subsection:\n### Community Engagement\n#### Volunteering for Housing Projects'}},
  {'name': 'Get Library Name Only',
   'step': 2,
   'action': 'LLM Prompt',
   'init': {'prompt': 'Outline Subsection:\n{outline sub section}\n\nThe information provided above uses the following format:\nLibrary Name: Libraryname\nTopic: Topic\nOutline Subsection:\n### Subsection\n#### Subsection Topic\n\nPlease extract the Library Name, and return ONLY the library name exactly as is!\n\nLibrary Name:',
    'as_list': False, 'split_on': None},
   'inputs': {'outline sub section': 'User Input - outline sub section'},
   'bubble_id': '1705855613936x209966971831189500',
   'output_type': None,
   'output': """LedgerwestgmailE49d4d23_4279_4fcf_ba76_727a53d9df1e""",
   'expected_input': {'outline sub section': 'Library Name: LedgerwestgmailE49d4d23_4279_4fcf_ba76_727a53d9df1e\nTopic: accessory-dwelling-unit-adu\nOutline Subsection:\n### Community Engagement\n#### Volunteering for Housing Projects'}},
  {'name': 'Subsection Research',
   'step': 3,
   'action': 'Library Research',
   'init': {'class_name': 'LedgerwestgmailLibrary from input',
    'k': 8,
    'as_qa': False,
    'from_similar_docs': False,
    'ignore_url': False,
    'split_on': None},
   'inputs': {'input': 'Get Outline Subsection Only',
    'Library From Input': 'Get Library Name Only'},
   'bubble_id': '1705855613936x273787311693758460',
   'output_type': None,
   'output': """Query:
The article reviews the best home improvement loans for December 2023.

Title: www.forbes.com_advisor_personal-loans_best-home-improvement-loans_
 Author: 
 Date: 2023-12-04
 URL: https://www.forbes.com/advisor/personal-loans/best-home-improvement-loans/
 loan rates depend on factors like your credit score, loan amount and repayment term. The lowest rates are typically reserved for the""",
   'expected_input': {
       'input': ["""Topic: accessory-dwelling-unit-adu
Outline Subsection:
### Community Engagement
#### Volunteering for Housing Projects"""],
       'Library From Input': """LedgerwestgmailE49d4d23_4279_4fcf_ba76_727a53d9df1e"""
    }},
  {'name': 'RenoFi-Provided External Sources',
   'step': 4,
   'action': 'Library Research',
   'init': {'class_name': 'LedgerwestgmailRenofiexternalsources',
    'k': 4,
    'as_qa': False,
    'from_similar_docs': False,
    'ignore_url': False,
    'split_on': None},
   'inputs': {'input': 'Get Outline Subsection Only'},
   'bubble_id': '1705855613936x221840173641498620',
   'output_type': None,
   'output': """Query:
Searching for RenoFi External Sources.

Title: www.forbes.com_advisor_personal-loans_best-home-improvement-loans_
 Author: 
 Date: 2023-12-04
 URL: https://www.forbes.com/advisor/personal-loans/best-home-improvement-loans/
 loan rates depend on factors like your credit score, loan amount and repayment term. The lowest rates are typically reserved for the""",
   'expected_input': {'input': ["""Topic: accessory-dwelling-unit-adu
Outline Subsection:
### Community Engagement
#### Volunteering for Housing Projects"""]}},
  {'name': 'Subsection Writer',
   'step': 5,
   'action': 'LLM Prompt',
   'init': {'prompt': 'External Research:\n{external research}\n\n{renofi-provided sources}\n\nTopic and Outline Subsection:\n{outline subsection}\n\nThe Topic and Outline Subsection provided above uses the following format:\nTopic: Topic (in format of all lowercase with hyphens and no special characters)\nOutline Subsection:\n### Subsection\n#### Subsection Topic\n\n\nYour Instructions:\nAbove, you\'ve been given relevant External Research, and a Topic and Outline Subsection. Your task here is to write a helpful and educational guide that uses the above information, and adheres to the following criteria:\n1) The Subsections and Subsection Topics provided in the Outline Subsection are suggestions, but feel free to adjust those titles to support the actual information provided above and used in the guide.\n2) Be sure the text in the guide is always in the context of the Topic and clearly declares its relationship to the Topic. For example, if a Subsection or Subsection topic is indirectly related to the Topic, the indirect relationship and relevance to the Topic should be made clear.\n3) It\'s important to use markdown hyperlinks like this, [anchor text](url), to cite External Research from above.  Always be sure to use the URL field exactly as provided when citing provided information.\n5) You represent RenoFi or renofi.com, so whenever you refer to RenoFi, you should do so in the first person (.e.g. we, us, here at RenoFi, etc...)\n6) Do not include any sources or content that contain foxscript.ai or that follow the pattern of Thisisaname_an09pzz_1sd, etc...\n7) It should be around 100 words long.\n8) Does this section compare two or more things? If so, try to put the comparison information into a table, so it\'s easy to read.  Use the HTML Table Syntax below instead of a markdown table. Ensure you use the div classes provided in the Syntax.\n\nTable Syntax:\n<div class="row mb-5 w-100">\n  <div class="table-responsive">\n    <table class="table table-responsive-sm">\n      <thead>\n          <tr>\n            <th class="swipe" scope="col"> </th>\n            <th scope="col">First column</th>\n            <th scope="col">Second column</th>\n            <th scope="col">Third column</th>\n          </tr>\n        </thead>\n        <tbody>\n          <tr>\n            <th scope="row">Typical Interest Rate</th>\n            <td>Market</td>\n            <td>Above Market</td>\n            <td>Above Market</td>\n          </tr>\n          <tr>\n            <th scope="row">Loan Limit</th>\n            <td>$500,000</td>\n            <td>Jumbos allowed</td>\n            <td>Conforming only</td>\n          </tr>\n        </tbody>\n      </table>\n  </div>\n</div>\n\n10) Don\'t use marketing hooks or sales pitches in the text!\n\n\nGuide:',
    'as_list': False, 'split_on': None},
   'inputs': {'external research': 'Subsection Research',
    'renofi-provided sources': 'RenoFi-Provided External Sources',
    'outline subsection': 'Get Outline Subsection Only'},
   'bubble_id': '1705855613936x515030616494309400',
   'output_type': None,
   'output': """The article reviews the best home improvement loans for December 2023, offering a comparison of various lenders based on APR ranges, loan amounts, and terms. Lenders like Best Egg, Discover, and LightStream are highlighted for their specific advantages such as secured loans, good-credit borrowers, and long-term loans, respectively. Additional lenders are discussed for military connections, short-term loans, same-day funding, joint loans, and fair- or bad-credit borrowers. The article emphasizes that personal loans for home improvements are unsecured, fixed-rate loans which can be a good choice for those who don't want to use their home equity but may come with higher interest rates for those with bad credit. It provides a guide on how to compare loans, the process of getting a home improvement loan, the average costs of home projects, and alternative financing options. The piece stresses it's crucial to consider APRs, loan amounts, terms, and fees when choosing a lender, and indicates that personal loan interest isn't tax-deductible, unlike home equity loans and HELOCs.""",
   'expected_input': {
       'external research': """Query:
The article reviews the best home improvement loans for December 2023.

Title: www.forbes.com_advisor_personal-loans_best-home-improvement-loans_
 Author: 
 Date: 2023-12-04
 URL: https://www.forbes.com/advisor/personal-loans/best-home-improvement-loans/
 loan rates depend on factors like your credit score, loan amount and repayment term. The lowest rates are typically reserved for the""",
       'renofi-provided sources': """Query:
Searching for RenoFi External Sources.

Title: www.forbes.com_advisor_personal-loans_best-home-improvement-loans_
 Author: 
 Date: 2023-12-04
 URL: https://www.forbes.com/advisor/personal-loans/best-home-improvement-loans/
 loan rates depend on factors like your credit score, loan amount and repayment term. The lowest rates are typically reserved for the""",
       'outline subsection': """Topic: accessory-dwelling-unit-adu
Outline Subsection:
### Community Engagement
#### Volunteering for Housing Projects"""
    }
  }
]}


workflow = Workflow().load_from_config(workflow_config, TEST_MODE=True)