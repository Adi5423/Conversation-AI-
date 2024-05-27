import spacy

nlp = spacy.load("en_core_web_sm")

def parse_sentence(sentence):
    doc = nlp(sentence)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    preps = [token.lemma_ for token in doc if token.pos_ == "PREP"]
    meaning = {
        "noun_phrases": noun_phrases,
        "verbs": verbs,
        "prepositions": preps,
    }
    return meaning

sentence = "Amelia explored the ancient ruins in the dense jungle. She climbed up the steep stone steps, admiring the intricate carvings on the walls. At the top, she gazed out over the vast canopy of trees, feeling a sense of awe. Later, she descended into the hidden chambers, illuminating the darkness with her flashlight. The mysterious symbols etched into the floor captured her attention, and she wondered about the civilization that once thrived here."
meaning = parse_sentence(sentence)
print(meaning)