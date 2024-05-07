import spacy

if __name__ == '__main__':
    nlp = spacy.load("zh_core_web_sm")
    sentence = nlp.tokenizer("我住在武汉")
    print("The number of tokens", len(sentence))
    print("The tokens: ")
    for words in sentence:
        print(words)