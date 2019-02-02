import math
class LaplaceTrigramLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.train(corpus)

    def train(self, corpus):
        """
        conditional probability p(w3|w1,w2) = c(w1,w2,w3)/c(w1,w2)
        append <r> </r> to the initial/final position of sentence for trigram
        """
        unigramCounts = {}
        bigramCounts = {}
        trigramCounts = {}
        for sentence in corpus.corpus:
            new_sentence = ['<r>'] + [i.word for i in sentence.data] + ['</r>']
            for i in range(len(new_sentence) - 2):  # -2 for the last trigram
                w1 = new_sentence[i]
                w2 = new_sentence[i+1]
                w3 = new_sentence[i+2]
                unigramCounts[w1] = unigramCounts.get(w1, 0) + 1
                bigramCounts[(w1, w2)] = bigramCounts.get((w1, w2), 0) + 1  # key is tuple (w1, w2)
                trigramCounts[(w1, w2, w3)] = trigramCounts.get((w1, w2, w3), 0) + 1  # key is tuple (w1, w2, w3)

            # for the last 2 index (count </s> </r>)
            s = new_sentence[-2]
            r = new_sentence[-1]
            unigramCounts[s] = unigramCounts.get(s, 0) + 1
            unigramCounts[r] = unigramCounts.get(r, 0) + 1
            bigramCounts[(s, r)] = bigramCounts.get((s, r), 0) + 1

        # save word count for add-one in the next test part
        self.unigram_count = unigramCounts
        self.bigram_count = bigramCounts
        self.trigram_count = trigramCounts

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

        ### calculate probability ###
        """
        p(w3|w1,w2) = c(w1,w2,w3)+1 / c(w1,w2)+V
        """
        # logP(W) = logP(<r>) + logP(<s>|<r>) + logP(w1|<r>,<s>) + logP(w2|<s>,w1) ...
        score = 0.0  # P(<r>) = P(<s>|<r>) = 1
        V = len(unigram_count)  # the number of words including UNK
        for i in range(2, len(sentence)):  # begin from the third index = logP(w1|<r>,<s>)
            w1 = sentence[i-2]
            w2 = sentence[i-1]
            w3 = sentence[i]
            cw1w2 = bigram_count[(w1, w2)] + V  # c(w1) + V
            cw1w2w3 = trigram_count[(w1, w2, w3)] + 1  # c(w1,w2) + 1
            prob = float(cw1w2w3 / cw1w2)  # calculate P(w3|w1,w2)
            score += math.log(prob)
        return score