# given log file, nobody can help.
# I don't know what data is in it, some sort of json.

import json 
from openai import OpenAI


file = {"cust_id":"A1023","email":"jane.doe@example.com","activity":"login","device":"mobile","geo":"US-NY","session_len":312,"credit_card":"4111-1111-1111-1234"}

model = ... # OpenAI(API=...)


for key, value in file.items():
    "####-####-####-####"
    ".*@.*..*"



for key, value in file.items():
    prompt = {
        "user": {f"Is this an email (answer yes/no only): {value}"},
        respond_json: False
    }

    
