from unittest_parametrize import param


input_fixture = {
    'params': ("init", "input", "check_input"),
    'values': [
        param(
            {
                'web_qa': False,
                'top_n': 1,
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
                'web_qa': False,
                'top_n': 1,
                'split_on': None
            }, {
                'input': 'https://www.cnn.com'
            }, {
                'input': ['https://www.cnn.com']
            },
            id='input2'
        ),
        param(
            {
                'web_qa': False,
                'top_n': 1,
                'split_on': '\n'
            }, {
                'input': 'https://www.cnn.com'
            }, {
                'input': ['https://www.cnn.com']
            },
            id='input3'
        ),
        param(
            {
                'web_qa': False,
                'top_n': 1,
                'split_on': '\n'
            }, {
                'input': 'best interest rate\nfastest loan'
            }, {
                'input': ['best interest rate', 'fastest loan']
            },
            id='input4'
        ),
        param(
            {
                'web_qa': False,
                'top_n': 1,
                'split_on': '<SPLIT>'
            }, {
                'input': 'best interest rate<SPLIT>fastest loan'
            }, {
                'input': ['best interest rate', 'fastest loan']
            },
            id='input5'
        )
    ]
}