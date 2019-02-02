import math
import numpy as np
class LaplaceBigramLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.train(corpus)

    def train(self, corpus):
        """
        Takes a corpus and trains your language model.Compute any counts or other corpus statistics in this function.
        """
        # TODO your code here
        # Tip: To get words from the corpus, try
        #    for sentence in corpus.corpus:
        #       for datum in sentence.data:
        #         word = datum.word
        """
        for bigram model, we need both unigram count c(w) and bigram count c(w1, w2)
        conditional probability p(w2|w1) = c(w1,w2)/c(w1)
        """
        unigramCounts = {}
        bigramCounts = {}
        for sentence in corpus.corpus:
            for i in range(len(sentence.data) - 1):  # -1 for the last bigram pair
                w1 = sentence.data[i].word
                w2 = sentence.data[i + 1].word
                unigramCounts[w1] = unigramCounts.get(w1, 0) + 1
                bigramCounts[(w1, w2)] = bigramCounts.get((w1, w2), 0) + 1  # key is tuple (w1, w2)

            # for the last index
            last_token = sentence.data[-1].word
            unigramCounts[last_token] = unigramCounts.get(last_token, 0) + 1

        # save word count and total for add-one in the next test part
        self.unigram_count = unigramCounts
        self.bigram_count = bigramCounts

    def score(self, sentence):
        """
        Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here

        # initialize count with trained data
        unigram_count = self.unigram_count.copy()
        bigram_count = self.bigram_count.copy()

        # make new key for UNK (unigram)
        for token in sentence:
            if token not in unigram_count:
              unigram_count[token] = 0

        # bigram_count without UNK > bigram_matrix with UNK (row = w1, column = w2)
        word_list = unigram_count.keys()  # get the list of all word including UNK
        word_to_index = {word: i for i, word in enumerate(word_list)}  #
        bigram_matrix = np.zeros((len(word_list), len(word_list)))
        for bigram, count in bigram_count.items():
            row_index = word_to_index[bigram[0]]
            column_index = word_to_index[bigram[1]]
            bigram_matrix[row_index][column_index] = count

        # add-one
        bigram_matrix += 1

        ### calculate probability ###
        """
        p(w2|w1) = c(w1,w2)/c(w1)  where c(w1) = sum(c(w1,*))
        bigram probability p(w2|w1) = c(w1,w2) / c(w1)
        but if we calculate all p(w2|w1) before, it takes a lot of time
        so calculate only p(w2|w1) that is needed after getting sentence
        """
        # logP(W) = logP(<s>) + logP(w1|<s>) + logP(w2|w1) + logP(w3|w2) ...
        score = 0.0  # P(<s>) = 1
        for i in range(1, len(sentence)):  # begin from the second index = logP(w1|<s>)
            w1 = sentence[i-1]
            w2 = sentence[i]
            cw1 = sum(bigram_matrix[word_to_index[w1]])  # c(w1) = sum(c(w1,*))
            cw1w2 = bigram_matrix[word_to_index[w1]][word_to_index[w2]]  # c(w1,w2)
            prob = float(cw1w2 / cw1)  # get P(word_i|word_i-1)
            score += math.log(prob)
        return score