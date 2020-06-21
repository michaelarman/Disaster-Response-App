from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def tokenize(text):
    '''
    Tokenizes the text field through word tokenizing, lemmatization and lowercasing
    INPUT:
    text - the message feature in the dataset
    OUTPUT:
    clean_tokens - the cleaned up text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens