import sys
sys.path.append('..')

from utils.workflow import Workflow
from .renofi_section import workflow as renofi_section_wf


workflow_config = {'name': 'V2 RenoFi Article',
 'steps': [{'name': 'Get Keyword',
   'step': 1,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Input:\n{input}\n\nFrom the JSON Input above, please return the Keyword value. Return only the Keyword value and nothing else!\n\nKeyword:',
    'as_list': False, 'split_on': None},
   'inputs': {'input': 'User Input - input'},
   'bubble_id': '1704998326900x188578676494827520',
   'output_type': None,
   'output': 'adu',
   'expected_input': {'input': '{"Date": "2024-01-19T13:46:43.4Z", "Hubpage": "ADU", "Keyword": "adu", "Volume": "46000", "Category": "Renovation Education", "Author": "RenoFi Team", "Editor": "Brian Powell", "Title": "What is an ADU (Accessory Dwelling Unit) ?", "Headline": "What is an ADU (Accessory Dwelling Unit) ?", "Slug": "accessory-dwelling-unit-adu"}'}},
  {'name': 'Get Slug',
   'step': 2,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Input:\n{input}\n\nFrom the JSON Input above, please return the Slug value. Return only the Slug value and nothing else! Do not wrap it in any additional characters, like \'\'\' or """.\n\nKeyword:',
    'as_list': False, 'split_on': None},
   'inputs': {'input': 'User Input - input'},
   'bubble_id': '1705606601260x497939957111062500',
   'output_type': None,
   'output': 'accessory-dwelling-unit-adu',
   'expected_input': {'input': '{"Date": "2024-01-19T13:46:43.4Z", "Hubpage": "ADU", "Keyword": "adu", "Volume": "46000", "Category": "Renovation Education", "Author": "RenoFi Team", "Editor": "Brian Powell", "Title": "What is an ADU (Accessory Dwelling Unit) ?", "Headline": "What is an ADU (Accessory Dwelling Unit) ?", "Slug": "accessory-dwelling-unit-adu"}'}},
  {'name': 'Get Subtopics',
   'step': 3,
   'user_id': '0x1',
   'action': 'Subtopics',
   'init': {'top_n': 10, 'by_source': False, 'split_on': None},
   'inputs': {'input': 'Get Keyword'},
   'bubble_id': '1703879018899x789331797976023000',
   'output_type': None,
   'output': """Subtopic 1
Sections of text found: 23
Theme: Nootropics and Adaptogens

Key Elements:
- Nootropics are natural substances that enhance cognitive performance and improve mental function by boosting memory, creativity, motivation, and attention.
- Nootropics work by boosting blood circulation and oxygenating the brain, reducing inflammation, protecting the brain from toxins, and stimulating the release of neurotransmitters.
- Neurotransmitters are chemical messengers released by neurons.
- Adaptogens are herbs and roots that support the body in handling mental and physical stress.
- Nootropics and adaptogens have different effects on the mind and body. Adaptogens help the body adapt to stress, while nootropics help the mind to adapt.
- Nootropics can be both synthetic and natural, while adaptogens are typically natural.
- Some adaptogens, such as Rhodiola, can also be classified as nootropics.
- B vitamins and certain plant and mushroom supplements can also have nootropic qualities.
- Nootropics and adaptogens can be used to improve cognitive function, memory, focus, creativity, and concentration.
- Nootropics can provide the brain with the conditions required to work at its maximum potential, while adaptogens work to balance the body's mental and physical reactions to stress.
- Research has shown that the supplementation of nootropics can significantly improve cognitive functioning and aid in the recovery of brain injury.

Sources:
- https://www.planetorganic.com/blogs/articles/differences-between-nootropics-and-adaptogens
- https://www.healingholidays.com/blog/a-guide-to-nootropics-and-adaptogens
- https://www.newchapter.com/wellness-blog/nutrition/adaptogens-vs-nootropics/
- https://www.glanbianutritionals.com/en/nutri-knowledge-center/insights/adaptogens-and-nootropics-future-mood-food-fortification
- https://ommushrooms.com/blogs/blog/adaptogens-and-nootropics-m2
- https://www.liveinnermost.com/blogs/insight/nootropics-and-adaptogens-the-low-down
- https://amass.com/blogs/impressions/what-are-adaptogens-and-nootropics

___
""",
   'expected_input': {'input': 'adu'}},
  {'name': 'Initial Outline',
   'step': 4,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Topic: {slug input}\n\nSubtopics:\n{subtopics}\n\nThe task here is to write an outline for an SEO content brief about the Topic above, using the Subtopics listed above. Organize the Subtopics into a complete and thorough logical outline.\n\nThe real challenge here is to organize the subtopics and information into a logical outline.\n\nAlso, return the list of unique URLs used in the Subtopics.  Return one URL per line and don\'t wrap it in """ or \'\'\' - just like in the template below.\n\nUse this outline template:\nUnique URLs:\n[list of unique urls, one per line]\n\nTopic: Topic from above (in format of all lowercase with hyphens and no special characters)\n\nI. Section\n   A. Subsection\n   1. Subsection Topic\n\nEvery Section must have at least one Subsection.  A Subsection can have Subsection Topics, but it doesn\'t have to. Do not include Introduction or Conclusion Sections in the Outline, those will be added later. Each Section should relate to the Topic above or be in the context of the Search Term above. \n\nOutline:',
    'as_list': False, 'split_on': None},
   'inputs': {'slug input': 'Get Slug', 'subtopics': 'Get Subtopics'},
   'bubble_id': '1703879018899x251102604985630720',
   'output_type': None,
   'output': """Unique URLs:
   https://www.url.com
   https://www.url2.com

   I. Introduction to Adaptogens and Nootropics
   A. Definition and Overview
      1. Explanation of Nootropics
      2. Explanation of Adaptogens
   B. The Growing Trend in Health and Wellness
      1. Popularity and Market Growth
      2. Consumer Demand Insights

II. Understanding Nootropics
   A. Cognitive Enhancement and Brain Health
      1. Mechanisms of Action
      2. Neurotransmitters and Their Roles
   B. Types of Nootropics
      1. Synthetic vs. Natural Nootropics
      2. Common Nootropic Supplements

III. Exploring Adaptogens
   A. Stress Response and Physical Adaptation
      1. Cortisol Regulation
      2. Immune and Endocrine System Support
   B. Types of Adaptogens
      1. Ashwagandha, Reishi, Rhodiola, and Others
      2. Adaptogenic Properties and Benefits""",
   'expected_input': {'slug input': 'accessory-dwelling-unit-adu', 'subtopics': "Subtopic 1\nSections of text found: 23\nTheme: Nootropics and Adaptogens\n\nKey Elements:\n- Nootropics are natural substances that enhance cognitive performance and improve mental function by boosting memory, creativity, motivation, and attention.\n- Nootropics work by boosting blood circulation and oxygenating the brain, reducing inflammation, protecting the brain from toxins, and stimulating the release of neurotransmitters.\n- Neurotransmitters are chemical messengers released by neurons.\n- Adaptogens are herbs and roots that support the body in handling mental and physical stress.\n- Nootropics and adaptogens have different effects on the mind and body. Adaptogens help the body adapt to stress, while nootropics help the mind to adapt.\n- Nootropics can be both synthetic and natural, while adaptogens are typically natural.\n- Some adaptogens, such as Rhodiola, can also be classified as nootropics.\n- B vitamins and certain plant and mushroom supplements can also have nootropic qualities.\n- Nootropics and adaptogens can be used to improve cognitive function, memory, focus, creativity, and concentration.\n- Nootropics can provide the brain with the conditions required to work at its maximum potential, while adaptogens work to balance the body's mental and physical reactions to stress.\n- Research has shown that the supplementation of nootropics can significantly improve cognitive functioning and aid in the recovery of brain injury.\n\nSources:\n- https://www.planetorganic.com/blogs/articles/differences-between-nootropics-and-adaptogens\n- https://www.healingholidays.com/blog/a-guide-to-nootropics-and-adaptogens\n- https://www.newchapter.com/wellness-blog/nutrition/adaptogens-vs-nootropics/\n- https://www.glanbianutritionals.com/en/nutri-knowledge-center/insights/adaptogens-and-nootropics-future-mood-food-fortification\n- https://ommushrooms.com/blogs/blog/adaptogens-and-nootropics-m2\n- https://www.liveinnermost.com/blogs/insight/nootropics-and-adaptogens-the-low-down\n- https://amass.com/blogs/impressions/what-are-adaptogens-and-nootropics\n\n___\n"}},
  {'name': 'Revised Outline',
   'step': 5,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': "Topic:\n{slug input}\n\nInitial Outline:\n{initial outline}\n\nPlease revise and streamline the initial outline above with the goal of removing redundant sections and information, as well as removing sections that seem completely unrelated to the Topic provided above.  You don't need to mention what you removed or anything.  Just return the revised outline only.  Also, you should ignore the list of URLs - don't include that in the Revised Outline.\n\nUse this outline template:\nTopic: Topic from above (in format of all lowercase with hyphens and no special characters)\n\nI. Section\n   A. Subsection\n   1. Subsection Topic\n\nEvery Section must have at least one Subsection.  A Subsection can have Subsection Topics, but it doesn't have to.  There may not be more than 10 Sections.  Do not include Introduction or Conclusion Sections in the Outline, those will be added later. Each Section should relate to the Topic above or be in the context of the Topic above. \n\nRevised Outline:",
    'as_list': False, 'split_on': None},
   'inputs': {'slug input': 'Get Slug', 'initial outline': 'Initial Outline'},
   'bubble_id': '1703879018899x563563707972976640',
   'output_type': None,
   'output': """I. Introduction to Adaptogens and Nootropics
   A. Definition and Overview
      1. Explanation of Nootropics
      2. Explanation of Adaptogens
   B. The Growing Trend in Health and Wellness
      1. Popularity and Market Growth
      2. Consumer Demand Insights

II. Understanding Nootropics
   A. Cognitive Enhancement and Brain Health
      1. Mechanisms of Action
      2. Neurotransmitters and Their Roles
   B. Types of Nootropics
      1. Synthetic vs. Natural Nootropics
      2. Common Nootropic Supplements""",
   'expected_input': {'slug input': 'accessory-dwelling-unit-adu', 'initial outline': 'Unique URLs:\n   https://www.url.com\n   https://www.url2.com\n\n   I. Introduction to Adaptogens and Nootropics\n   A. Definition and Overview\n      1. Explanation of Nootropics\n      2. Explanation of Adaptogens\n   B. The Growing Trend in Health and Wellness\n      1. Popularity and Market Growth\n      2. Consumer Demand Insights\n\nII. Understanding Nootropics\n   A. Cognitive Enhancement and Brain Health\n      1. Mechanisms of Action\n      2. Neurotransmitters and Their Roles\n   B. Types of Nootropics\n      1. Synthetic vs. Natural Nootropics\n      2. Common Nootropic Supplements\n\nIII. Exploring Adaptogens\n   A. Stress Response and Physical Adaptation\n      1. Cortisol Regulation\n      2. Immune and Endocrine System Support\n   B. Types of Adaptogens\n      1. Ashwagandha, Reishi, Rhodiola, and Others\n      2. Adaptogenic Properties and Benefits'}},
  {'name': 'Get Subtopic URLs',
   'step': 6,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Initial Outline:\n{initial outline}\n\nFrom the Initial Outline above, please return only the list of Unique URLs provided at the very top.  Return them exactly as-is, one url per line.  Don\'t wrap them in anything like """ or \'\'\'\n\nUnique URLs:',
    'as_list': False, 'split_on': None},
   'inputs': {'initial outline': 'Initial Outline'},
   'bubble_id': '1703879018899x881013720042176500',
   'output_type': None,
   'output': """https://www.url.com
https://www.url2.com""",
   'expected_input': {'initial outline': 'Unique URLs:\n   https://www.url.com\n   https://www.url2.com\n\n   I. Introduction to Adaptogens and Nootropics\n   A. Definition and Overview\n      1. Explanation of Nootropics\n      2. Explanation of Adaptogens\n   B. The Growing Trend in Health and Wellness\n      1. Popularity and Market Growth\n      2. Consumer Demand Insights\n\nII. Understanding Nootropics\n   A. Cognitive Enhancement and Brain Health\n      1. Mechanisms of Action\n      2. Neurotransmitters and Their Roles\n   B. Types of Nootropics\n      1. Synthetic vs. Natural Nootropics\n      2. Common Nootropic Supplements\n\nIII. Exploring Adaptogens\n   A. Stress Response and Physical Adaptation\n      1. Cortisol Regulation\n      2. Immune and Endocrine System Support\n   B. Types of Adaptogens\n      1. Ashwagandha, Reishi, Rhodiola, and Others\n      2. Adaptogenic Properties and Benefits'}},
  {'name': 'Make Subtopics Library',
   'step': 7,
   'user_id': '0x1',
   'action': 'Send Output',
   'init': {'destination': 'Workflow Library',
    'as_workflow_doc': False,
    'target_doc_input': False,
    'as_url_list': True,
    'empty_doc': False,
    'csv_doc': False,
    'delimiter': ',',
    'drive_folder': 'root',
    'to_rtf': False,
    'split_on': None},
   'inputs': {'input': 'Get Subtopic URLs'},
   'bubble_id': '1703879018899x362353205038546940',
   'output_type': None,
   'output': """LedgerwestgmailE5029ae5_540b_4660_84b8_e479b52bc3ec""",
   'expected_input': {'input': 'https://www.url.com\nhttps://www.url2.com'}},
  {'name': 'Draft Document',
   'step': 8,
   'user_id': '0x1',
   'action': 'Send Output',
   'init': {'destination': 'Project',
    'as_workflow_doc': True,
    'target_doc_input': False,
    'as_url_list': False,
    'empty_doc': True,
    'csv_doc': False,
    'delimiter': ',',
    'drive_folder': 'root',
    'to_rtf': False,
    'split_on': None},
   'inputs': {'input': 'Get Slug'},
   'bubble_id': '1703879018899x859639536211722200',
   'output_type': None,
   'output': """1705081617636x357237695061795460""",
   'expected_input': {'input': 'accessory-dwelling-unit-adu'}},
  {'name': 'Get Outline Sections',
   'step': 9,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Library Name: {subtopics library}\nDraft Document ID: {draft document}\n\nContent Brief Outline:\n{content brief outline}\n\nThe Content Brief Outline provided above uses the following format:\nTopic: Topic \n\nI. Section\n   A. Subsection\n   1. Subsection Topic\n\nFrom the provided Content Brief Outline, please return each of the Sections along with their Subsections and Subsection Topics. At the top of the Section include the Library Name, the Draft Document ID, and the Topic provided above, and at the end of the Section insert <SPLIT> to mark the end, like the example below.  Replace the Outline roman numerals, letters, and numbers with Markdown headers like the example below.\n\nLibrary Name: [library name]\nDraft Document ID: [draft document ID]\nTopic: [Topic] \nOutline Section:\n## Section\n   ### Subsection\n   #### Subsection Topic\n   ### Subsection\n<SPLIT>\n\nList of Sections:',
    'as_list': False, 'split_on': None},
   'inputs': {'subtopics library': 'Make Subtopics Library',
    'draft document': 'Draft Document',
    'content brief outline': 'Revised Outline'},
   'bubble_id': '1703879018899x499012818166874100',
   'output_type': None,
   'output': """Draft Document ID: 1705081617636x357237695061795460
Library Name: LedgerwestgmailE5029ae5_540b_4660_84b8_e479b52bc3ec
Topic: Private Credit vs. Private Equity
Outline Section:
## What is Private Credit?
<SPLIT>
Draft Document ID: 1705081617636x357237695061795460
Library Name: LedgerwestgmailE5029ae5_540b_4660_84b8_e479b52bc3ec
Topic: Private Credit vs. Private Equity
Outline Section:
## What is Private Equity?
<SPLIT>
Draft Document ID: 1705081617636x357237695061795460
Library Name: LedgerwestgmailE5029ae5_540b_4660_84b8_e479b52bc3ec
Topic: Private Credit vs. Private Equity
Outline Section:
## The Similarities and Differences
""",
   'expected_input': {'subtopics library': 'LedgerwestgmailE5029ae5_540b_4660_84b8_e479b52bc3ec', 'draft document': '1705081617636x357237695061795460', 'content brief outline': 'I. Introduction to Adaptogens and Nootropics\n   A. Definition and Overview\n      1. Explanation of Nootropics\n      2. Explanation of Adaptogens\n   B. The Growing Trend in Health and Wellness\n      1. Popularity and Market Growth\n      2. Consumer Demand Insights\n\nII. Understanding Nootropics\n   A. Cognitive Enhancement and Brain Health\n      1. Mechanisms of Action\n      2. Neurotransmitters and Their Roles\n   B. Types of Nootropics\n      1. Synthetic vs. Natural Nootropics\n      2. Common Nootropic Supplements'}},
  {'name': 'Write Sections',
   'step': 10,
   'user_id': '0x1',
   'action': 'Workflow',
   'init': {'workflow': renofi_section_wf,
    'in_parallel': False,
    'split_on': "<SPLIT>"},
   'inputs': {'input': 'Get Outline Sections'},
   'bubble_id': '1703879018899x311159217143939100',
   'output_type': None,
   'output': """I. Introduction to Adaptogens and Nootropics

### A. Definition and Overview

Adaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.

### 1. Explanation of Nootropics

Nootropics, often referred to as "smart drugs" or cognitive enhancers, are substances that aim to improve mental functions such as memory, creativity, motivation, and attention. These compounds work through various mechanisms, including boosting blood circulation to the brain, enhancing oxygen utilization, reducing inflammation, and stimulating the release of neurotransmitters, which are the brain's chemical messengers. The use of nootropics has been associated with improved cognitive functioning and may even aid in the recovery of brain injuries. They can be found in both synthetic forms and as natural supplements, with natural options often having additional health benefits and a lower risk of side effects. [Planet Organic](https://www.planetorganic.com/blogs/articles/differences-between-nootropics-and-adaptogens), [Healing Holidays](https://www.healingholidays.com/blog/a-guide-to-nootropics-and-adaptogens)""",
   'expected_input': {'User Input - outline section': ['Draft Document ID: 1705081617636x357237695061795460\nLibrary Name: LedgerwestgmailE5029ae5_540b_4660_84b8_e479b52bc3ec\nTopic: Private Credit vs. Private Equity\nOutline Section:\n## What is Private Credit?\n', '\nDraft Document ID: 1705081617636x357237695061795460\nLibrary Name: LedgerwestgmailE5029ae5_540b_4660_84b8_e479b52bc3ec\nTopic: Private Credit vs. Private Equity\nOutline Section:\n## What is Private Equity?\n', '\nDraft Document ID: 1705081617636x357237695061795460\nLibrary Name: LedgerwestgmailE5029ae5_540b_4660_84b8_e479b52bc3ec\nTopic: Private Credit vs. Private Equity\nOutline Section:\n## The Similarities and Differences\n']}},
  {'name': 'Get Main Article Content',
   'step': 11,
   'user_id': '0x1',
   'action': 'Fetch Input',
   'init': {'source': 'Document From Input', 'split_on': None},
   'inputs': {'input': 'Draft Document'},
   'bubble_id': '1703879018899x295916516192288800',
   'output_type': None,
   'output': """I. Introduction to Adaptogens and Nootropics

### A. Definition and Overview

Adaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.

### 1. Explanation of Nootropics

Nootropics, often referred to as "smart drugs" or cognitive enhancers, are substances that aim to improve mental functions such as memory, creativity, motivation, and attention. These compounds work through various mechanisms, including boosting blood circulation to the brain, enhancing oxygen utilization, reducing inflammation, and stimulating the release of neurotransmitters, which are the brain's chemical messengers. The use of nootropics has been associated with improved cognitive functioning and may even aid in the recovery of brain injuries. They can be found in both synthetic forms and as natural supplements, with natural options often having additional health benefits and a lower risk of side effects. [Planet Organic](https://www.planetorganic.com/blogs/articles/differences-between-nootropics-and-adaptogens), [Healing Holidays](https://www.healingholidays.com/blog/a-guide-to-nootropics-and-adaptogens)""",
   'expected_input': {'input': '1705081617636x357237695061795460'}},
  {'name': 'Write Intro',
   'step': 12,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Main Article Content:\n{article}\n\nWrite an Introduction for the Main Article Content provided above and give the Introduction a relevant Heading Title.  It should follow the same basic style, voice, and format as the Main Article Content. Return only the new Introduction for the Article and nothing else, like the example below. The Introduction should begin with the {{< table-of-contents >}} element at the very beginning exactly as is (like in the example below). The Introduction should end with the exact CTA provided below.\n\nCTA:\n<div class="d-flex">\n  {{{{<cta-link target="_blank" cta="See Rates" location="{slug}">}}}}\n</div>\n\n\nIntroduction Example:\n{{{{< table-of-contents >}}}}\n\n## Relevant Intro Heading\nIntroduction content here...\n\nCTA here\n\n\nIntroduction:',
    'as_list': False, 'split_on': None},
   'inputs': {'article': 'Get Main Article Content', 'slug': 'Get Slug'},
   'bubble_id': '1703880021266x691413030024511500',
   'output_type': None,
   'output': """Introduction

Adaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.""",
   'expected_input': {'article': 'I. Introduction to Adaptogens and Nootropics\n\n### A. Definition and Overview\n\nAdaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.\n\n### 1. Explanation of Nootropics\n\nNootropics, often referred to as "smart drugs" or cognitive enhancers, are substances that aim to improve mental functions such as memory, creativity, motivation, and attention. These compounds work through various mechanisms, including boosting blood circulation to the brain, enhancing oxygen utilization, reducing inflammation, and stimulating the release of neurotransmitters, which are the brain\'s chemical messengers. The use of nootropics has been associated with improved cognitive functioning and may even aid in the recovery of brain injuries. They can be found in both synthetic forms and as natural supplements, with natural options often having additional health benefits and a lower risk of side effects. [Planet Organic](https://www.planetorganic.com/blogs/articles/differences-between-nootropics-and-adaptogens), [Healing Holidays](https://www.healingholidays.com/blog/a-guide-to-nootropics-and-adaptogens)', 'slug': 'accessory-dwelling-unit-adu'}},
{'name': 'Write Conclusion',
   'step': 13,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Main Article Content:\n{article}\n\nWrite a Conclusion for the Main Article Content provided above and give the Conclusion a relevant Heading title.  It should follow the same basic style, voice, and format as the Main Article Content. Return only the new Conclusion for the Article and nothing else.\n\n\nConclusion Example:\n## Relevant Conclusion Heading\nConclusion content here...\n\n\nConclusion:',
    'as_list': False, 'split_on': None},
   'inputs': {'article': 'Get Main Article Content'},
   'bubble_id': '1703880201863x763258102474604500',
   'output_type': None,
   'output': """Conclusion

Adaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.""",
   'expected_input': {'article': 'I. Introduction to Adaptogens and Nootropics\n\n### A. Definition and Overview\n\nAdaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.\n\n### 1. Explanation of Nootropics\n\nNootropics, often referred to as "smart drugs" or cognitive enhancers, are substances that aim to improve mental functions such as memory, creativity, motivation, and attention. These compounds work through various mechanisms, including boosting blood circulation to the brain, enhancing oxygen utilization, reducing inflammation, and stimulating the release of neurotransmitters, which are the brain\'s chemical messengers. The use of nootropics has been associated with improved cognitive functioning and may even aid in the recovery of brain injuries. They can be found in both synthetic forms and as natural supplements, with natural options often having additional health benefits and a lower risk of side effects. [Planet Organic](https://www.planetorganic.com/blogs/articles/differences-between-nootropics-and-adaptogens), [Healing Holidays](https://www.healingholidays.com/blog/a-guide-to-nootropics-and-adaptogens)'}},
  {'name': 'Intro + Article',
   'step': 14,
   'user_id': '0x1',
   'action': 'Combine',
   'init': {},
   'inputs': {'First Item': 'Write Intro',
    'Second Item': 'Get Main Article Content'},
   'bubble_id': '1703880262579x702460956800188400',
   'output_type': None,
   'output': """Introduction

Adaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.

More Content""",
   'expected_input': {'First Item': 'Introduction\n\nAdaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.', 'Second Item': 'I. Introduction to Adaptogens and Nootropics\n\n### A. Definition and Overview\n\nAdaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.\n\n### 1. Explanation of Nootropics\n\nNootropics, often referred to as "smart drugs" or cognitive enhancers, are substances that aim to improve mental functions such as memory, creativity, motivation, and attention. These compounds work through various mechanisms, including boosting blood circulation to the brain, enhancing oxygen utilization, reducing inflammation, and stimulating the release of neurotransmitters, which are the brain\'s chemical messengers. The use of nootropics has been associated with improved cognitive functioning and may even aid in the recovery of brain injuries. They can be found in both synthetic forms and as natural supplements, with natural options often having additional health benefits and a lower risk of side effects. [Planet Organic](https://www.planetorganic.com/blogs/articles/differences-between-nootropics-and-adaptogens), [Healing Holidays](https://www.healingholidays.com/blog/a-guide-to-nootropics-and-adaptogens)'}},
  {'name': 'Intro + Article + Conclusion',
   'step': 15,
   'user_id': '0x1',
   'action': 'Combine',
   'init': {},
   'inputs': {'First Item': 'Intro + Article',
    'Second Item': 'Write Conclusion'},
   'bubble_id': '1703880296473x448643993269174300',
   'output_type': None,
   'output': """Introduction

Adaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.

More Content

Conclusion.""",
   'expected_input': {'First Item': 'Introduction\n\nAdaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.\n\nMore Content', 'Second Item': 'Conclusion\n\nAdaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.'}},
  {'name': 'Get Description',
   'step': 16,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Article Introduction:\n{introduction}\n\nPlease write a one-sentence summary description of the Introduction provided above. This will be used as metadata for the Article.\n\nDescription:',
    'as_list': False, 'split_on': None},
   'inputs': {'introduction': 'Write Intro'},
   'bubble_id': '1705607173066x208404268929515520',
   'output_type': None,
   'output': """Short meta description of article.""",
   'expected_input': {'introduction': 'Introduction\n\nAdaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.'}},
  {'name': 'Metadata Header',
   'step': 17,
   'user_id': '0x1',
   'action': 'LLM Prompt',
   'init': {'prompt': 'Input:\n{input}\n\nDescription: {description}\n\nMetadata Template:\n---\ntitle: Title\nheadline: Headline\ndescription: Description\ndate: Date\ntype: learn-post\ndraft: false\neditor: Editor\nauthors:\n  - Authors\nslug: Slug\nsubcategory: Sub Category\ntags:\n  - Home Improvement Loan\n  - Hubpage\n  - Keyword\n---\n\nThe Input above is in JSON format. Using the Input and Metadata Template provided above, please map the Input values to the Metadata values according the Metadata Template and return the Metadata. If a Metadata value isn\'t present in the Input, just leave it blank in the Metadata. You must maintain the exact format of the Metadata Template above, beginning with --- and ending with ---. Do not wrap the result in any additional text, like \'\'\' or """.  Do not insert new lines. Return in the exact format as above.\n\nMetadata:',
    'as_list': False, 'split_on': None},
   'inputs': {'input': 'User Input - input', 'description': 'Get Description'},
   'bubble_id': '1704998992913x265088851032145900',
   'output_type': None,
   'output': """---
title: Adu Austin, TX
headline: Adu Austin, TX
description: This guide provides an overview of various home renovation financing options, helping homeowners understand and navigate the financial aspects of upgrading their living spaces.
date: 2024-01-18T11:43:26.1Z
draft: false
editor: Brian Powell
authors:
  - RenoFi Team
slug: adu-austin-tx
tags:
  - "Home Improvement Loan"
  - ADU
  - Renovation Education
  - adu austin
---""",
   'expected_input': {'input': '{"Date": "2024-01-19T13:46:43.4Z", "Hubpage": "ADU", "Keyword": "adu", "Volume": "46000", "Category": "Renovation Education", "Author": "RenoFi Team", "Editor": "Brian Powell", "Title": "What is an ADU (Accessory Dwelling Unit) ?", "Headline": "What is an ADU (Accessory Dwelling Unit) ?", "Slug": "accessory-dwelling-unit-adu"}', 'description': 'Short meta description of article.'}},
  {'name': 'Metadata + Article',
   'step': 18,
   'user_id': '0x1',
   'action': 'Combine',
   'init': {},
   'inputs': {'First Item': 'Metadata Header',
    'Second Item': 'Intro + Article + Conclusion'},
   'bubble_id': '1704999032726x794869481805447200',
   'output_type': None,
   'output': """---
title: Adu Austin, TX
headline: Adu Austin, TX
description: This guide provides an overview of various home renovation financing options, helping homeowners understand and navigate the financial aspects of upgrading their living spaces.
date: 2024-01-18T11:43:26.1Z
draft: false
editor: Brian Powell
authors:
  - RenoFi Team
slug: adu-austin-tx
tags:
  - "Home Improvement Loan"
  - ADU
  - Renovation Education
  - adu austin
---

Article Stuff...""",
   'expected_input': {'First Item': '---\ntitle: Adu Austin, TX\nheadline: Adu Austin, TX\ndescription: This guide provides an overview of various home renovation financing options, helping homeowners understand and navigate the financial aspects of upgrading their living spaces.\ndate: 2024-01-18T11:43:26.1Z\ndraft: false\neditor: Brian Powell\nauthors:\n  - RenoFi Team\nslug: adu-austin-tx\ntags:\n  - "Home Improvement Loan"\n  - ADU\n  - Renovation Education\n  - adu austin\n---', 'Second Item': 'Introduction\n\nAdaptogens and nootropics are two categories of substances that have gained popularity in the health and wellness industry for their potential benefits to mental and physical well-being. Adaptogens are natural herbs and roots known for their ability to help the body resist and adapt to stress, while nootropics are substances that may enhance cognitive function and brain health. Both have unique properties and mechanisms of action, and they can be used separately or together to support overall health.\n\nMore Content\n\nConclusion.'}},
  {'name': 'Ensure Relative RenoFi Links',
   'step': 19,
   'user_id': '0x1',
   'action': 'Run Code',
   'init': {'py_code': "output = input.replace('https://renofi.com','').replace('https://www.renofi.com','')",
    'code_from_input': False},
   'inputs': {'input': 'Metadata + Article'},
   'bubble_id': '1706200944395x340265405554098200',
   'output_type': None,
   'output': """---
title: Adu Austin, TX
headline: Adu Austin, TX
description: This guide provides an overview of various home renovation financing options, helping homeowners understand and navigate the financial aspects of upgrading their living spaces.
date: 2024-01-18T11:43:26.1Z
draft: false
editor: Brian Powell
authors:
  - RenoFi Team
slug: adu-austin-tx
tags:
  - "Home Improvement Loan"
  - ADU
  - Renovation Education
  - adu austin
---

Article Stuff with relative links.""",
   'expected_input': {'input': '---\ntitle: Adu Austin, TX\nheadline: Adu Austin, TX\ndescription: This guide provides an overview of various home renovation financing options, helping homeowners understand and navigate the financial aspects of upgrading their living spaces.\ndate: 2024-01-18T11:43:26.1Z\ndraft: false\neditor: Brian Powell\nauthors:\n  - RenoFi Team\nslug: adu-austin-tx\ntags:\n  - "Home Improvement Loan"\n  - ADU\n  - Renovation Education\n  - adu austin\n---\n\nArticle Stuff...'}},
  {'name': 'Save To Drive',
   'step': 20,
   'user_id': '0x1',
   'action': 'Send Output',
   'init': {'destination': 'Google Drive',
    'as_workflow_doc': False,
    'target_doc_input': False,
    'as_url_list': False,
    'empty_doc': False,
    'csv_doc': False,
    'delimiter': ',',
    'drive_folder': '1W49nVSnqrN4afRAUs8Tv9A5HHeEBQcPR',
    'to_rtf': False,
    'split_on': None},
   'inputs': {'input': 'Ensure Relative RenoFi Links', 'Title': 'Get Slug'},
   'bubble_id': '1704999068442x752412135931248600',
   'output_type': None,
   'output': """000x111xasdfasdf""",
   'expected_input': {'input': '---\ntitle: Adu Austin, TX\nheadline: Adu Austin, TX\ndescription: This guide provides an overview of various home renovation financing options, helping homeowners understand and navigate the financial aspects of upgrading their living spaces.\ndate: 2024-01-18T11:43:26.1Z\ndraft: false\neditor: Brian Powell\nauthors:\n  - RenoFi Team\nslug: adu-austin-tx\ntags:\n  - "Home Improvement Loan"\n  - ADU\n  - Renovation Education\n  - adu austin\n---\n\nArticle Stuff with relative links.', 'Title': 'accessory-dwelling-unit-adu'}}]}


workflow = Workflow().load_from_config(workflow_config, TEST_MODE=True)