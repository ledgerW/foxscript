from unittest_parametrize import param


input_fixture = {
    'params': ("init", "input", "check_input"),
    'values': [
        param(
            {
                'class_name': 'LedgerwestgmailSitecontent',
                'k': 1,
                'as_qa': False,
                'from_similar_docs': False,
                'ignore_url': False,
                'split_on': None
            }, {
                'input': 'best interest rate'
            }, {
                'input': ['best interest rate']
            },
            id='input1'
        ),
        param(
            {
                'class_name': 'LedgerwestgmailSitecontent',
                'k': 1,
                'as_qa': False,
                'from_similar_docs': False,
                'ignore_url': False,
                'split_on': ''
            }, {
                'input': 'best interest rate'
            }, {
                'input': ['best interest rate']
            },
            id='input2'
        ),
        param(
            {
                'class_name': 'LedgerwestgmailSitecontent',
                'k': 1,
                'as_qa': False,
                'from_similar_docs': False,
                'ignore_url': False,
                'split_on': '\n'
            }, {
                'input': 'best interest rate\nfastest loan'
            }, {
                'input': ['best interest rate', 'fastest loan']
            },
            id='input3'
        ),
        param(
            {
                'class_name': 'LedgerwestgmailSitecontent',
                'k': 1,
                'as_qa': False,
                'from_similar_docs': False,
                'ignore_url': False,
                'split_on': '<SPLIT>'
            }, {
                'input': 'best interest rate<SPLIT>fastest loan'
            }, {
                'input': ['best interest rate', 'fastest loan']
            },
            id='input4'
        )
    ]
}