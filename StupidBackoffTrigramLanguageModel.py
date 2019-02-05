import math
class StupidBackoffTrigramLanguageModel:

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
        trigramCounts = {}
        total = 0
        for sentence in corpus.corpus:
            new_sentence = ['<r>'] + [i.word for i in sentence.data] + ['</r>']  # start tag for trigram
            for i in range(len(new_sentence) - 2):  # -2 for the last trigram
                w1 = new_sentence[i]
                w2 = new_sentence[i + 1]
                w3 = new_sentence[i + 2]
                unigramCounts[w1] = unigramCounts.get(w1, 0) + 1
                bigramCounts[(w1, w2)] = bigramCounts.get((w1, w2), 0) + 1  # key is tuple (w1, w2)
                trigramCounts[(w1, w2, w3)] = trigramCounts.get((w1, w2, w3), 0) + 1  # key is tuple (w1, w2, w3)
                total += 1

            # for the last 2 index (count </s> </r>)
            s = new_sentence[-2]
            r = new_sentence[-1]
            unigramCounts[s] = unigramCounts.get(s, 0) + 1
            unigramCounts[r] = unigramCounts.get(r, 0) + 1
            bigramCounts[(s, r)] = bigramCounts.get((s, r), 0) + 1

        # save word count and total for next test part
        self.unigram_count = unigramCounts
        self.bigram_count = bigramCounts
        self.trigram_count = trigramCounts
        self.total = total

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here

        # initialize count with trained data
        unigram_count = self.unigram_count.copy()
        bigram_count = self.bigram_count.copy()
        trigram_count = self.trigram_count.copy()
        N = self.total

        # total vocab number including UNK for laplace smoothing
        V = len(self.unigram_count.keys() | set(sentence))

        # start tag for trigram
        sentence = ['<r>'] + sentence + ['</r>']

        ### calculate probability ###
        """
        if key (w1,w2,w3) is in trigram
        S(w3|w1,w2) = c(w1,w2,w3) / c(w1,w2) ... normal trigram
        
        elif key (w2,w3) is in bigram
        S(w3|w1,w2) = k * c(w2,w3) / c(w2) ... k * normal bigram
        
        elif key (w3) is in unigram
        S(w3|w1,w2) = k * c(w3) / (N + V) ... k * laplace unigram (if k^2 lower score)
        
        else ... w3 = UNK
        S(w3|w1,w2) = k * 1 / (N + V) ... k * laplace unigram
        """
        # logP(W) = logP(<r>) + logP(<s>|<r>) + logP(w1|<r>,<s>) + logP(w2|<s>,w1) ...
        score = 0.0  # P(<r>) = P(<s>|<r>) = 1
        k = 0.8  # coefficient for stupid backoff
        for i in range(2, len(sentence)):  # begin from the third index = logP(w1|<r>,<s>)
            w1 = sentence[i-2]
            w2 = sentence[i-1]
            w3 = sentence[i]
            if (w1, w2, w3) in trigram_count:
                S = trigram_count[(w1, w2, w3)] / bigram_count[(w1, w2)]  # normal trigram
            elif (w2, w3) in bigram_count:
                S = k * bigram_count[(w2, w3)] / unigram_count[w2]  # k * normal bigram
            elif w3 in unigram_count:
                S = k * (unigram_count[w3]+1) / (N + V)  # k * Laplace unigram
            else:
                S = k * 1 / (N + V)  # k * Laplace unigram
            score += math.log(S)

        return score