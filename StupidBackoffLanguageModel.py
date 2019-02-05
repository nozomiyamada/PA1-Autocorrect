import math
class StupidBackoffLanguageModel:

    def __init__(self, corpus):
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
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
        total = 0
        for sentence in corpus.corpus:
            for i in range(len(sentence.data) - 1):  # -1 for the last bigram pair
                w1 = sentence.data[i].word
                w2 = sentence.data[i + 1].word
                unigramCounts[w1] = unigramCounts.get(w1, 0) + 1
                bigramCounts[(w1, w2)] = bigramCounts.get((w1, w2), 0) + 1  # key is tuple (w1, w2)
                total += 1

            # for the last index (count </s> for P(</s>))
            last_token = sentence.data[-1].word
            unigramCounts[last_token] = unigramCounts.get(last_token, 0) + 1

        # save word count and total for next test part
        self.unigram_count = unigramCounts
        self.bigram_count = bigramCounts
        self.total = total

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here
        unigram_count = self.unigram_count.copy()
        bigram_count = self.bigram_count.copy()
        N = self.total

        # total vocab number including UNK for laplace smoothing
        V = len(self.unigram_count.keys() | set(sentence))

        ### calculate probability ###
        """
        if key (w1,w2) is in bigram
        S(w2|w1) = c(w1,w2) / c(w1) ... normal bigram
        
        elif key w2 is in unigram ... k * unsmoothed unigram
        S(w2|w1) = k * c(w1) / N
        
        else ... w2 = UNK ... k * laplace unigram
        S(w2|w1) = k * 1 / (N + V)
        """
        # logP(W) = logP(<s>) + logP(w1|<s>) + logP(w2|w1) + logP(w3|w2) ...
        score = 0.0  # P(<s>) = 1
        k = 0.4  # coefficient for stupid backoff
        for i in range(1, len(sentence)):  # begin from the second index = logP(w1|<s>)
            w1 = sentence[i-1]
            w2 = sentence[i]
            if (w1, w2) in bigram_count:
                cw1 = unigram_count[w1]
                cw1w2 = bigram_count[(w1, w2)]
                S = cw1w2 / cw1  # normal bigram P(wi|wi-1)
            elif w2 in unigram_count:
                S = k * (unigram_count[w2] + 1) / (N + V)  # k * laplace unigram
            else:
                S = k * 1 / (N + V)  # k * laplace unigram
            score += math.log(S)

        return score