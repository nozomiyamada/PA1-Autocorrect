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

            # for the last index (count </s> for P(</s>))
            last_token = sentence.data[-1].word
            unigramCounts[last_token] = unigramCounts.get(last_token, 0) + 1

        # save word count and for add-one in the next test part
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

        # total vocab number including UNK for laplace smoothing
        V = len(unigram_count.keys() | set(sentence))

        # make new key for UNK (unigram)
        for word in sentence:
            if word not in unigram_count:
                unigram_count[word] = 0

        # make new key for UNK (bigram)
        for i in range(1, len(sentence)):
            bigram = (sentence[i-1], sentence[i])
            if bigram not in bigram_count:
                bigram_count[bigram] = 0

        ### calculate probability ###
        """
        p(w2|w1) = c(w1,w2) + 1 / c(w1) + V
        but if we calculate all p(w2|w1) before, it takes a lot of time
        so calculate only p(w2|w1) that is needed after getting sentence
        """
        # logP(W) = logP(<s>) + logP(w1|<s>) + logP(w2|w1) + logP(w3|w2) ...
        score = 0.0  # P(<s>) = 1
        for i in range(1, len(sentence)):  # begin from the second index = logP(w1|<s>)
            w1 = sentence[i-1]
            w2 = sentence[i]
            cw1 = unigram_count[w1] + V  # c(w1) + V
            cw1w2 = bigram_count[(w1, w2)] + 1  # c(w1,w2) + 1
            prob = cw1w2 / cw1  # P(wi|wi-1) = c(wi-1, wi) + 1 / c(wi) + V
            score += math.log(prob)
        return score