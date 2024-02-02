from unittest_parametrize import param

from fixtures.workflows import (
    renofi_article_wf,
    renofi_section_wf,
    renofi_subsection_wf,
    child_memo_highlights_wf,
    memo_highlights_wf
)


io_fixture = {
    'params': ("workflow", "input_var", "input_val"),
    'values': [
        param(
            renofi_article_wf,
            ['User Input - input'],
            ['{"Date": "2024-01-19T13:46:43.4Z", "Hubpage": "ADU", "Keyword": "adu", "Volume": "46000", "Category": "Renovation Education", "Author": "RenoFi Team", "Editor": "Brian Powell", "Title": "What is an ADU (Accessory Dwelling Unit) ?", "Headline": "What is an ADU (Accessory Dwelling Unit) ?", "Slug": "accessory-dwelling-unit-adu"}'],
            id='renofi_article_wf'
        ),
        param(
            renofi_subsection_wf,
            ['User Input - outline sub section'],
            ['Library Name: LedgerwestgmailE49d4d23_4279_4fcf_ba76_727a53d9df1e\nTopic: accessory-dwelling-unit-adu\nOutline Subsection:\n### Community Engagement\n#### Volunteering for Housing Projects'],
            id='renofi_subsection_wf'
        ),
        param(
            child_memo_highlights_wf,
            ['User Input - input'],
            ['https://www.url.com'],
            id='child_memo_highlights_wf'
        ),
        param(
            memo_highlights_wf,
            ['User Input - urls'],
            ['https://www.url.com\nhttps://www.url2.com'],
            id='memo_highlights_wf'
        )
    ]
}