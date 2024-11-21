from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def tokenize(text):
    """
    Tokenizes the given text into a list of tokens.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: The list of tokens.
    """
    return word_tokenize(text)

def detokenize(tokens):
    """
    Detokenizes a list of tokens into a single text.

    Args:
        tokens (list): The list of tokens to detokenize.

    Returns:
        str: The detokenized text.
    """
    return TreebankWordDetokenizer().detokenize(tokens)

def segment(text):
    """
    Segments the given text into a list of sentences.

    Args:
        text (str): The text to segment.

    Returns:
        list: The list of segmented sentences.
    """
    return sent_tokenize(text)