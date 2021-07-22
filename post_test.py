# -*- coding: utf-8 -*-
import sys
import requests
import json
import random

def test_json_prediction(host='http://127.0.0.1:5000/', data=False):
    print('test_json_prediction...')
    if not data:
        pass
    #print(json.dumps(data, indent=4, sort_keys=True))
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    url = host+"predict/json"
    response = requests.post(url, json=data, headers=headers, timeout=3600.001)
    obj = json.loads(response.content)
    print(obj['prediction'])


if __name__ == "__main__":

    # sample data for request
    frames = [
        ['5575-GNVDE', 'Male', 0.0, 'No', 'No', 34.0, 'Yes', 'No', 'DSL', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'One year', 'No', 'Mailed check', 56.95, 1889.5],
        ['3668-QPYBK', 'Male', 0.0, 'No', 'No', 2.0, 'Yes', 'No', 'DSL', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Month-to-month', 'Yes', 'Mailed check', 53.85, 108.15],
        ['7795-CFOCW', 'Male', 0.0, 'No', 'No', 45.0, 'No', 'No phone service', 'DSL', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'One year', 'No', 'Bank transfer (automatic)', 42.3, 1840.75]
    ]

    json_to_send = {}
    json_to_send['input_frames'] = frames
    test_json_prediction(data=json_to_send)
