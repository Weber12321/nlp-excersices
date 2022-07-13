from collections import Counter, namedtuple

from interface.interface import MethodInterface


class Ngram(MethodInterface):
    def __init__(self, n=2):
        super().__init__()
        self.n = n

    def ngram(self):
        grams = []
        words = []

        for sentence in self.data:
            sentence = ['<s>'] + list(sentence) + ['</s>']
            for i in range(len(sentence)-self.n+1):
                grams.append(tuple(sentence[i:i+self.n]))
            for i in range(len(sentence)-self.n+2):
                words.append(tuple(sentence[i:i+self.n-1]))

        total_word_counter = Counter(grams)
        word_counter = Counter(words)

        return total_word_counter, word_counter

    def predict(self):
        pred = {}
        Word = namedtuple('Word', ['word', 'prob'])

        total_word_counter, word_counter = self.ngram()
        for k in total_word_counter:
            word = ''.join(k[:self.n-1])
            if word not in pred:
                pred.update({word: set()})

            next_word_prob = total_word_counter[k] / word_counter[k[:self.n-1]]
            w = Word(k[-1], '{:.3g}'.format(next_word_prob))
            pred[word].add(w)

        for word, n_gram in pred.items():
            pred[word] = sorted(n_gram, key=lambda x: x.prob, reverse=True)

        return pred

    def run(self):
        return self.predict()
