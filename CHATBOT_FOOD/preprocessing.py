import nltk, re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from tensorflow import keras
import re
import os

stop_words = stopwords.words("english")
normalizer = WordNetLemmatizer()


def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len(
        [item for item in probable_part_of_speech if item.pos() == "n"]
    )
    pos_counts["v"] = len(
        [item for item in probable_part_of_speech if item.pos() == "v"]
    )
    pos_counts["a"] = len(
        [item for item in probable_part_of_speech if item.pos() == "a"]
    )
    pos_counts["r"] = len(
        [item for item in probable_part_of_speech if item.pos() == "r"]
    )
    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech


def preprocess_text(text):
    cleaned = re.sub(r"\W+", " ", text).lower()
    tokenized = word_tokenize(cleaned)
    normalized = [
        normalizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized
    ]
    return normalized


stop_words = set(stopwords.words("english"))


def preprocess(input_sentence):
    input_sentence = input_sentence.lower()
    input_sentence = re.sub(r"[^\w\s]", "", input_sentence)
    tokens = word_tokenize(input_sentence)
    input_sentence = [i for i in tokens if not i in stop_words]
    return input_sentence


def compare_overlap(user_message, possible_response):
    similar_words = 0
    for token in user_message:
        if token in possible_response:
            similar_words += 1
    return similar_words


def extract_nouns(tagged_message):
    message_nouns = list()
    for token in tagged_message:
        if token[1].startswith("N"):
            message_nouns.append(token[0])
    return message_nouns


def compute_similarity(tokens, category):
    output_list = list()
    for token in tokens:
        output_list.append([token.text, category.text, token.similarity(category)])
    return output_list
