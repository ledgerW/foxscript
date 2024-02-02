from unittest_parametrize import param


input_fixture = {
    'params': ("init", "input", "check_input", "check_prompt"),
    'values': [
        param(
            {
                'prompt': "Check prompt with {i1}",
                'as_list': False
            }, {
                'i1': 'i1 text'
            }, {
                'i1': 'i1 text'
            },
            'Check prompt with i1 text',
            id='input1'
        ),
        param(
            {
                'prompt': "Check prompt with {i1} {i2}",
                'as_list': False
            }, {
                'i1': 'i1 text',
                'i2': 'i2 text'
            }, {
                'i1': 'i1 text',
                'i2': 'i2 text'
            },
            'Check prompt with i1 text i2 text',
            id='input2'
        ),
        param(
            {
                'prompt': "Check prompt with {i1} {i2} {i3}",
                'as_list': False
            }, {
                'i1': 'i1 text',
                'i2': 'i2 text',
                'i3': 'i3 text'
            }, {
                'i1': 'i1 text',
                'i2': 'i2 text',
                'i3': 'i3 text'
            },
            'Check prompt with i1 text i2 text i3 text',
            id='input3'
        ),
        param(
            {
                'prompt': "Check prompt with {i1} {i2} {i3} {i4}",
                'as_list': False
            }, {
                'i1': 'i1 text',
                'i2': 'i2 text',
                'i3': 'i3 text',
                'i4': 'i4 text'
            }, {
                'i1': 'i1 text',
                'i2': 'i2 text',
                'i3': 'i3 text',
                'i4': 'i4 text'
            },
            'Check prompt with i1 text i2 text i3 text i4 text',
            id='input4'
        )
    ]
}

call_fixture = {
    'params': ("init", "input"),
    'values': [
        param(
            {
                'prompt': """1. item 1

                From the above list, please return item number {number} only, extactly as-is, with nothing else.

                Item:
                """,
                'as_list': False
            }, {
                'number': '1'
            },
            id='input1'
        )
    ]
}