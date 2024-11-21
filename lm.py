from corpus import detokenize, tokenize, segment 
from collections import defaultdict
import random
 

class LanguageModel:
    
    def __init__(self, n):
        self.n = n
        self.counts = {}
        self.vocabulary = {}        

    def train(self, file):
        """
        Trains the language model using the text from a file.

        Args:
            file (str): The path to the text file used for training.
        """
        with open(file) as f:
            self.text = f.read()
        #f.close()
        sentences = segment(self.text)  
        if self.n > 2:
            self.text = ' None ' * (self.n - 2)
        else:
            self.text = ''

        for sent in sentences:
            self.text += ' None ' + sent + ' None '

        if self.n > 2:
            self.text += ' None ' * (self.n - 2)

        self.tokens = tokenize(self.text)
        self.n_grams = [tuple(self.tokens[i: i + self.n]) for i in range(len(self.tokens) - self.n + 1)]
        self.vocabulary = set(self.n_grams)


        self.stats = {}
        for ngram in self.n_grams:
            if ngram[:-1] not in self.stats:
                self.stats[ngram[:-1]] = {'CNT': 1, ngram[-1]: {'CNT': 1}}
            elif ngram[-1] not in self.stats[ngram[:-1]]:
                self.stats[ngram[:-1]]['CNT'] += 1
                self.stats[ngram[:-1]][ngram[-1]] = {'CNT': 1}
            else:
                self.stats[ngram[:-1]]['CNT'] += 1
                self.stats[ngram[:-1]][ngram[-1]]['CNT'] += 1

    
        for ngram, dict in self.stats.items():
            for token in dict:
                if token == 'CNT':
                    continue
                self.stats[ngram][token]['FRQ'] = \
                    self.stats[ngram][token]['CNT'] / self.stats[ngram]['CNT']
        self.count()
    
    def count(self):
        """
        Counts the occurrences of n-grams and calculates probabilities.

        Raises:
            ValueError: Not possible for n = 1.
        """
        if self.n == 1:
            raise ValueError("Counting is not possible for n = 1.")

        mainlist = []
        prob_dict = {}
        
        for i in self.n_grams:
            mainlist.append(tuple(i[:self.n-1]))
        random_dict = defaultdict(lambda: defaultdict(int))
        mainlist = set(mainlist)
        for i in self.n_grams:
            prefix = tuple(i[:self.n-1])
            next_word = i[self.n-1]
            if prefix in mainlist:
                random_dict[prefix][next_word] += 1
        self.counts = random_dict
        for i in random_dict.keys():
            prob_dict[i] = self.normalize(i)
        self.probs = prob_dict
        
        
    def normalize(self, token):

        """
        Normalizes the counts of a token to probabilities.

        Args:
            token (tuple): The token (n-1 gram) to normalize.

        Returns:
            list: The list of probabilities corresponding to each possible next tok.
        """
        cts = self.counts[token]
        acc = 0
        probs = []
        for i in cts:
            acc += cts[i]
        for i in cts:
            probs.append(cts[i]/acc)
        return probs
    
    def sample(self, token):
        """
        Samples next token based on the given token.

        Args:
            token (tuple): The token (n-1 gram) to sample from.

        Returns:
            str: The sampled next token.
        """
        unit = random.random()
        a = list(self.counts[token].keys())
        num_line = []
        acc = 0
        probs = self.probs[token]
        for i in probs:
            acc += i
            num_line.append(acc)
        num_line.append(1)
        for num, i in enumerate(num_line):
            if unit < i:
                return a[num]
        return a[-1]
            

    def p_next(self, token_seq):
        """
        Calculates the probabilities of all possible next tokens given a token sequence.

        Args:
            token_seq (tuple): The token sequence (n-1 gram) for which to calculate probabilities.

        Returns:
            list: The list of tuples containing each possible next token and its corresponding probability.
        """
        prob_d = []
        for token, dict in self.stats[tuple(token_seq)].items():
            if token == 'CNT':
                continue
            prob_d.append(tuple([token,
                                 self.stats[tuple(token_seq)][token]['FRQ']]))


        return prob_d


    def generate(self):
        """
        Generates a new string based on the model.

        Returns:
            str: The generated string.
        """
        tokens = []
        history = ('None',) * (self.n - 1)
        while True:
            next_word = self.sample(history)
            if next_word == 'None':
                break
            tokens.append(next_word)
            history = history[1:] + (next_word,)
        return detokenize(tokens)

    def write_to_file(self, text, filename, num_texts, ns):
        """
        Writes generated texts to a file.

        Args:
            text (str): The text to be written to the file.
            filename (str): The name of the file to write to.
            num_texts (int): The number of texts user decides.
            ns (int): The number of paragraphs to write.
        """
        with open(filename, 'w') as f:
            for _ in range(num_texts):
                text = ' '.join(self.generate() for _ in range(ns))
                text = text.replace(' .', '.')
                f.write(text + '\n')
                num_texts -= 1
                if num_texts == 0:
                    break 
        print(f'\nTexts written to {filename}.')