import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import langdetect
open_json = "Dataset//pre_process//Final_json//tokenization//"
list_name = ['Anushka_Mishra','Arsh','Bandaid','Chuwii','Double_battery','Naina','Nikita','Whookid_Africa']

# Load the JSON file
with open(f'{open_json}tokenized_Anushka_Mishra.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create a list of responses and their corresponding tokens
english_responses = []
hinglish_responses = []
english_tokens_list = []
hinglish_tokens_list = []

hinglish_keywords = ['ka', 'ki', 'ko', 'kya', 'hai', 'hain', 'ho', 'ja', 'ji', 'jo','yrrrr','yaar','ha','haan','han']  # Add more keywords as needed

for item in data:
    response = item['response']
    tokens = ' '.join(item['tokens'])
    
    # Check if the response contains any Hinglish keywords
    if any(keyword in tokens.lower() for keyword in hinglish_keywords):
        hinglish_responses.append(response)
        hinglish_tokens_list.append(tokens)
    else:
        try:
            language = langdetect.detect(response)
            if language == 'en':  # English
                english_responses.append(response)
                english_tokens_list.append(tokens)
        except langdetect.lang_detect_exception.LangDetectException:
            print(f"Skipping response: {response} (language detection failed)")
            continue

# Create a custom list of stop words for Hinglish (WhatsApp language)
hinglish_stop_words = ['thn', 'thnk', 'u', 'ur', 'yr', 'pls', 'plz', 'thx', 'tnx', 'btw', 'brb', 'b4', 'afta', 'wit', 'witout', 'hv', 'hw', 'y', 'r', 'v', 'm', 'n', 'jst', 'jus', 'wt', 'wtvr', 'w8', 'cud', 'shud', 'wud', 'cnt', 'cn', 'cd', 'wd', 'll', 'vl', 'vll', 'fyn', 'f9', 'gr8', 'l8', 'tym', 'tm', 'tod', 'ystrdy', 'yrstrdy']

# Create a BoW representation using CountVectorizer for English text
english_vectorizer = CountVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
english_bow_matrix = english_vectorizer.fit_transform(english_tokens_list)

# Create a BoW representation using CountVectorizer for Hinglish text
hinglish_vectorizer = CountVectorizer(stop_words=hinglish_stop_words, max_features=1000, ngram_range=(1, 2))
hinglish_bow_matrix = hinglish_vectorizer.fit_transform(hinglish_tokens_list)

# Create a TF-IDF representation using TfidfVectorizer for English text
english_tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
english_tfidf_matrix = english_tfidf_vectorizer.fit_transform(english_tokens_list)

# Create a TF-IDF representation using TfidfVectorizer for Hinglish text
hinglish_tfidf_vectorizer = TfidfVectorizer(stop_words=hinglish_stop_words, max_features=1000, ngram_range=(1, 2))
hinglish_tfidf_matrix = hinglish_tfidf_vectorizer.fit_transform(hinglish_tokens_list)

# Print the BoW and TF-IDF matrices
print("English BoW Matrix:")
print(english_bow_matrix.toarray())
print("\nEnglish TF-IDF Matrix:")
print(english_tfidf_matrix.toarray())
print("\nHinglish BoW Matrix:")
print(hinglish_bow_matrix.toarray())
print("\nHinglish TF-IDF Matrix:")
print(hinglish_tfidf_matrix.toarray())