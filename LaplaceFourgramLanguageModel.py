import math
class LaplaceFourgramLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.train(corpus)

    def train(self, corpus):
        """
        conditional probability p(w4|w1,w2,w3) = c(w1,w2,w3,w4)/c(w1,w2,w3)
        append <q> <r> </r> </q> to the initial/final position of sentence for 4gram
        """
        unigramCounts = {}
        bigramCounts = {}
        trigramCounts = {}
        fourgramCounts = {}
        for sentence in corpus.corpus:
            new_sentence = ['<q>', '<r>'] + [i.word for i in sentence.data] + ['</r>', '</q>']
            for i in range(len(new_sentence) - 3):  # -3 for the last 4gram
                w1 = new_sentence[i]
                w2 = new_sentence[i+1]
                w3 = new_sentence[i+2]
                w4 = new_sentence[i+3]
                unigramCounts[w1] = unigramCounts.get(w1, 0) + 1
                bigramCounts[(w1, w2)] = bigramCounts.get((w1, w2), 0) + 1  # key is tuple (w1, w2)
                trigramCounts[(w1, w2, w3)] = trigramCounts.get((w1, w2, w3), 0) + 1
                fourgramCounts[(w1, w2, w3, w4)] = fourgramCounts.get((w1, w2, w3, w4), 0) + 1

            # for the last 3 index (count </s> </r> </q>)
            s = new_sentence[-3]
            r = new_sentence[-2]
            q = new_sentence[-1]
            unigramCounts[s] = unigramCounts.get(s, 0) + 1
            unigramCounts[r] = unigramCounts.get(r, 0) + 1
            unigramCounts[q] = unigramCounts.get(q, 0) + 1
            bigramCounts[(s, r)] = bigramCounts.get((s, r), 0) + 1
            bigramCounts[(r, q)] = bigramCounts.get((r, q), 0) + 1
            trigramCounts[(s, r, q)] = trigramCounts.get((s, r, q), 0) + 1

        # save word count for add-one in the next test part
        self.unigram_count = unigramCounts
        self.bigram_count = bigramCounts
        self.trigram_count = trigramCounts
        self.fourgram_count = fourgramCounts

    def score(self, sentence):
        """
        Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here

        # initialize count with trained data
        unigram_count = self.unigram_count.copy()
        bigram_count = self.bigram_count.copy()
        trigram_count = self.trigram_count.copy()
        fourgram_count = self.fourgram_count.copy()

        # make new key for UNK (unigram)
        for token in sentence:
            if token not in unigram_count:
                unigram_count[token] = 0

        # make new key for UNK (bigram)
        for i in range(1, len(sentence)):
            bigram = (sentence[i-1], sentence[i])
            if bigram not in bigram_count:
                bigram_count[bigram] = 0

        # make new key for UNK (trigram)
        for i in range(2, len(sentence)):
            trigram = (sentence[i-2], sentence[i-1], sentence[i])
            if trigram not in trigram_count:
                trigram_count[trigram] = 0

        # make new key for UNK (4gram)
        for i in range(3, len(sentence)):
            fourgram = (sentence[i-3], sentence[i-2], sentence[i-1], sentence[i])
            if fourgram not in fourgram_count:
                fourgram_count[fourgram] = 0

        ### calculate probability ###
        """
        p(w4|w1,w2,w3) = c(w1,w2,w3,w4)+1 / c(w1,w2,w3)+V
        """
        # logP(W) = logP(<q>) + logP(<r>|<q>) + logP(<s>|<q>,<r>) + logP(w1|<q>,<r>,<s>) ...
        score = 0.0  # P(<q>) = P(<r>|<q>) = P(<s>|<q>,<r>) = 1
        V = len(unigram_count)  # the number of words including UNK
        for i in range(3, len(sentence)):  # begin from the fourth index = P(w1|<q>,<r>,<s>)
            w1 = sentence[i-3]
            w2 = sentence[i-2]
            w3 = sentence[i-1]
            w4 = sentence[i]
            cw1w2w3 = trigram_count[(w1, w2, w3)] + V  # c(w1,w2,w3) + V
            cw1w2w3w4 = fourgram_count[(w1, w2, w3, w4)] + 1  # c(w1,w2,w3,w4) + 1
            prob = float(cw1w2w3w4 / cw1w2w3)  # calculate P(w4|w1,w2,w3,w4)
            score += math.log(prob)

        return score