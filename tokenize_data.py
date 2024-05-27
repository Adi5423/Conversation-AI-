import json
from transformers import AutoTokenizer

location = "Dataset//pre_process//Final_json//"

# Load the JSON file
with open(location+"Whookid_Africa.json", "r",encoding = "utf-8") as input_file:
    data = json.load(input_file)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
tokenizer.add_special_tokens({'additional_special_tokens': ['😊', '🫶🏻','🫀','🤌🏻', '🤔','😂','🥹','🫠','😭','💌','🫂','🥱','😴']})

# Tokenize the responses
for item in data:
    response = item["response"]
    tokens = tokenizer.tokenize(response)
    item["tokens"] = tokens

# Save the tokenized data to a new JSON file
with open("Dataset//pre_process//Final_json//tokenization//tokenized_Whookid_Africa.json", "w",encoding = "utf-8") as output_file:
    json.dump(data, output_file, indent=2)