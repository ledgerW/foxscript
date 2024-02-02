import sys
sys.path.append('..')

from utils.workflow import Workflow
from .renofi_subsection import workflow as renofi_subsection_wf


workflow_config = {'name': 'V2 RenoFi Section',
 'steps': [{'name': 'Get Outline Subsections',
   'step': 1,
   'action': 'LLM Prompt',
   'init': {'prompt': 'Provided:\n{outline section}\n\nThe information provided above uses the following format:\nLibrary Name: Libraryname\nDraft Document ID: DraftdocumentID\nTopic: Topic (in format of all lowercase with hyphens and no special characters)\nOutline Section:\n## Section\n   ### Subsection\n   #### Subsection Topic\n   ### Subsection\n\nFrom the provided Outline Section above, please return each of the Subsections along with their Subsection Topics (if any). At the top of the Subsection include the Library Name and Topic, and at the end of the Subsection insert <SPLIT> to mark the end, like the example below.  You can ignore the Draft Document ID.\n\nLibrary Name: Libraryname\nTopic: Topic (in format of all lowercase with hyphens and no special characters)\nOutline Subsection:\n### Subsection\n#### Subsection Topic\n<SPLIT>\n\nList of Subsections:',
    'as_list': True},
   'inputs': {'outline section': 'User Input - outline section'},
   'bubble_id': '1705855672504x581815232403865600',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Write Subsections',
   'step': 2,
   'action': 'Workflow',
   'init': {'workflow': renofi_subsection_wf,
    'in_parallel': True},
   'inputs': {'input': 'Get Outline Subsections'},
   'bubble_id': '1705855672504x999717045453717500',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Get Section Name',
   'step': 3,
   'action': 'LLM Prompt',
   'init': {'prompt': 'Provided:\n{outline section}\n\nThe information provided above uses the following format:\nLibrary Name: Libraryname\nDraft Document ID: DraftdocumentID\nTopic: Topic (in format of all lowercase with hyphens and no special characters)\nOutline Section:\n## Section\n   ### Subsection\n   #### Subsection Topic\n   ### Subsection\n\nPlease return the Section Name from the content Provided above.  For example, ## Section Name, or ## Other Section Name.\n\nSection Name:',
    'as_list': False},
   'inputs': {'outline section': 'User Input - outline section'},
   'bubble_id': '1705855672504x150838502652641280',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Append Section Above Subsections',
   'step': 4,
   'action': 'Combine',
   'init': {},
   'inputs': {'First Item': 'Get Section Name',
    'Second Item': 'Write Subsections'},
   'bubble_id': '1705855672504x445862789787156500',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Get Draft ID',
   'step': 5,
   'action': 'LLM Prompt',
   'init': {'prompt': 'Outline Subsection:\n{outline sub section}\n\nThe information provided above uses the following format:\nLibrary Name: Libraryname\nDraft Document ID: DraftdocumentID\nTopic: Topic\nOutline Section:\n## Section\n   ### Subsection\n   #### Subsection Topic\n   ### Subsection\n\nPlease extract the Draft Document ID, and return ONLY the Draft Document ID as is!\n\nDraft Document ID:',
    'as_list': False},
   'inputs': {'outline sub section': 'User Input - outline section'},
   'bubble_id': '1705855672504x760170467683991600',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Get Running Draft',
   'step': 6,
   'action': 'Fetch Input',
   'init': {'source': 'Document From Input'},
   'inputs': {'input': 'Get Draft ID'},
   'bubble_id': '1705855672504x747721697051279400',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Get Topic',
   'step': 7,
   'action': 'LLM Prompt',
   'init': {'prompt': 'Provided:\n{outline section}\n\nThe information provided above uses the following format:\nLibrary Name: Libraryname\nDraft Document ID: DraftdocumentID\nTopic: Topic (in format of all lowercase with hyphens and no special characters)\nOutline Section:\n## Section\n   ### Subsection\n   #### Subsection Topic\n   ### Subsection\n\nPlease return the Topic from the content Provided above, and only the Topic. Return it in the format of all lowercase with hyphens and no special characters.\n\nTopic:',
    'as_list': False},
   'inputs': {'outline section': 'User Input - outline section'},
   'bubble_id': '1705855672504x457784281939050500',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Edited Section',
   'step': 8,
   'action': 'LLM Prompt',
   'init': {'prompt': 'Unfinished Article:\n{unfinished article}\n\nNext Section Draft:\n{next section draft}\n\nAbove, you\'ve been provided with what we have so far of an Unfinished Article we plan to publish, as well as the initial draft of the Next Section in the Article. Your job is to do a final edit of the Next Section Draft found above with the goal of...\na) Eliminate repetitive phrases. Our readers hate it when we repeat ourselves.\nb) Don\'t remove any Tables of information.\nc) Ensure the whole Article has logical transitions from one Section and Subsection to the next.\nd) Referencing sources of information with the provided url link (preferred) or at least mentioning the source.\ne) The Headers and Subheaders in the Next Section Draft are suggestions, but feel free to adjust those titles to support the actual information provided above and used in the Draft.\nf) Be sure the text in the guide is always in the context of the Unfinished Article Topic and clearly declares its relationship to the Topic. For example, if a section of text is indirectly related to the Topic, the indirect relationship and relevance to the Topic should be made clear.\n\nOnly return the edited Next section of the Article - DO NOT RETURN THE WHOLE ARTICLE.\n\nDon\'t forget to adhere to our style guide below!\n\nStyle Guide:\n## Tone and Voice\n- **Professional and informative:** The writing should convey expertise and confidence in the subject matter.\n- **Friendly and approachable:** Use a conversational tone that is welcoming to readers who may not be familiar with financial products.\n- **Clear and concise:** Avoid jargon and complex language. Explain financial concepts in simple terms.\n\n## Audience\n- Homeowners considering renovation projects.\n- Individuals seeking information on home renovation financing options.\n- Readers with varying levels of financial literacy.\n\n## Language and Grammar\n- **Active Voice:** Use active voice to create a direct and engaging narrative.\n- **Second Person:** Address the reader as "you" to create a personal connection.\n- **Present Tense:** Generally, use present tense to describe ongoing situations and current options.\n- **Consistency:** Maintain consistent verb tenses and terminology throughout the article.\n\n## Branding\n- **RenoFi Brand:** Maintain a positive and supportive tone when discussing RenoFi products and services. RenoFi is a financial technology company that provides a lending platform leveraged by credit unions nationwide to offer RenoFi Loans to consumers. Currently, RenoFi Loan products include HELOC and fixed home equity loans. RenoFi Loan products are intended for consumers looking for non-traditional loans to support their renovation, remodeling, and/or construction financing needs. The RenoFi team is made up of expert loan originators who are dedicated to providing expert information and personalized loan concierge services via RenoFi financial technology.\n\n## Markdown Formatting\nBold: Use **bold** to call out key names or elements\nItalic: Use *italic* to emphasize a specific word\nHyperlink: Use [url slug as text](url) to cite sources, like this: [Slug Text](https://domain.com/slug-text). When using a hyperlink, put it in a sentence, but don\'t use it inside parenthesis.\nStrikethrough: ~~scratch this.~~\nBlock Quote: > Use blockquotes to highlight a direct quote\nLists:  Use lists like below.\n1. First ordered list item\n1. Another item\n   - Unordered sub-list.\n   - Unordered sub-list 2.\n1. Actual numbers don\'t matter, just that it\'s a number\n   1. Ordered sub-list\n   1. Actual numbers don\'t matter\n\n## Special Notes\n- You represent RenoFi or renofi.com, so whenever you refer to RenoFi, you should do so in the first person (.e.g. we, us, here at Renofi, etc...)\n- Don\'t use marketing hook and sales pitch language in the text! Only use the CTA that is described below.\n- Hyperlinks should not be used inside of parentheses.\n\n## Call to Action (CTA)\n- If and only if it flows well with the Section, include the CTA below to encourage readers to take the next step. But DO NOT use a CTA if the previous Section has a CTA. You must use the exact text and syntax of the CTA provided below, including the <div> tags. Don\'t change anything about the CTA and don\'t use a hyperlink instead.\n\nCTA:\n<div class="d-flex">\n  {{{{<cta-link target="_blank" cta="See Rates" location="{topic}">}}}}\n</div>\n\n\nEdited Next Section:',
    'as_list': False},
   'inputs': {'unfinished article': 'Get Running Draft',
    'next section draft': 'Append Section Above Subsections',
    'topic': 'Get Topic'},
   'bubble_id': '1705855672504x381894599141752800',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Get Section Outline Only',
   'step': 9,
   'action': 'LLM Prompt',
   'init': {'prompt': "Provided:\n{outline section}\n\nThe information provided above uses the following format:\nLibrary Name: Libraryname\nDraft Document ID: DraftdocumentID\nTopic: Topic (in format of all lowercase with hyphens and no special characters)\nOutline Section:\n## Section\n   ### Subsection\n   #### Subsection Topic\n   ### Subsection\n\nFrom the provided Outline Section above, please return the Topic and Outline Section only. You can ignore the Library Name and Draft Document ID. Don't add anything to the output. Below is an example of what to return.\n\nExample:\nTopic: Topic (in format of all lowercase with hyphens and no special characters)\n## Section\n   ### Subsection\n   #### Subsection Topic\n   ### Subsection\n\n\nTopic and Outline Section Only:",
    'as_list': False},
   'inputs': {'outline section': 'User Input - outline section'},
   'bubble_id': '1705857769381x960397505530953700',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Renofi Internal Pages',
   'step': 10,
   'action': 'Library Research',
   'init': {'class_name': 'LedgerwestgmailRenofiproductpages',
    'k': 3,
    'as_qa': False,
    'from_similar_docs': True,
    'ignore_url': False},
   'inputs': {'input': 'Get Section Outline Only'},
   'bubble_id': '1705857534733x981151064583045100',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Final Section',
   'step': 11,
   'action': 'LLM Prompt',
   'init': {'prompt': "Edited Draft:\n{edited draft}\n\n\nInstructions:\nAbove, you've been provided with an Edited Draft that is nearly ready to publish. The only thing left to do is weave in 1 internal link to existing related articles we've already published. The most important thing about an internal link is the anchor text - a) it must be tied directly to the text in the article that it links to, and b) it should be similar to the url slug.\n\nYour job is to insert just one internal link with perfect anchor text into the Edited Draft provided above. Below, you've been provided with a few Raw Internal Link Options to build the internal link. Pick one of them to use for the internal link, then insert it into the Edited Draft.\n\nRaw Internal Link Options:\n{renofi internal pages}\n\n\nThe Raw Internal Link Options above are in the following format:\nQuery:\nquery content...\n\nTitle: title\nAuthor: author or BLANK\nDate: date\nURL: full url\ntext from article...\n\n\nUse markdown hyperlinks to create the internal link, like this: [anchor text](url).\n\n- Very Important! The anchor text for the link should be just a few words that capture the nature of the text from the article or be similar to the url slug.\n\nReturn the Edited Draft with the new Internal Link inserted in a relevant location, and nothing else. Other than the new internal link, the Edited Draft should not change. Definitely do not include the Raw Internal Link Options in the new Edited Draft.\n\n\nNew Edited Draft with Internal Link:",
    'as_list': False},
   'inputs': {'edited draft': 'Edited Section',
    'renofi internal pages': 'Renofi Internal Pages'},
   'bubble_id': '1705858530544x419580475931885600',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Draft With New Section',
   'step': 12,
   'action': 'Combine',
   'init': {},
   'inputs': {'First Item': 'Get Running Draft',
    'Second Item': 'Final Section'},
   'bubble_id': '1705855672504x107636427251777540',
   'output_type': None,
   'output': '',
   'expected_input': ''},
  {'name': 'Save Updated Article',
   'step': 13,
   'action': 'Send Output',
   'init': {'destination': 'Project',
    'as_workflow_doc': True,
    'target_doc_input': True,
    'as_url_list': False,
    'empty_doc': False,
    'csv_doc': False,
    'delimiter': ',',
    'drive_folder': 'root',
    'to_rtf': False},
   'inputs': {'input': 'Draft With New Section', 'Target Doc': 'Get Draft ID'},
   'bubble_id': '1705855672504x524071266008956900',
   'output_type': None,
   'output': '',
   'expected_input': ''}]}


workflow = Workflow().load_from_config(workflow_config, TEST_MODE=True)