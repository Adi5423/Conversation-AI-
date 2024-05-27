import json

location_json = "Dataset//pre_process//Final_json//"
location = "Dataset//pre_process//"

def preprocess_conversation(conversation_text):
    conversations = []
    user = ""
    response = ""

    lines = conversation_text.strip().split("\n")
    for line in lines:
        if ":" in line:
            index = line.index(":")
            user_line = line[:index].strip()
            response_line = line[index + 1:].strip()

            if user:
                conversations.append({"user": user, "response": response})
                user = ""
                response = response_line
            else:
                user = user_line
                response = response_line
        else:
            response += " " + line.strip()

    if user and response:
        conversations.append({"user": user, "response": response})

    return conversations

def save_conversations_to_json(conversations, file_name):
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(conversations, json_file, ensure_ascii=False, indent=4)

with open(location+"output_Arohii.txt", "r", encoding="utf-8") as file:
    conversation_text = file.read()

conversations = preprocess_conversation(conversation_text)
save_conversations_to_json(conversations,f"{location_json}conversation_Arohii.json")