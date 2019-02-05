import math
class CustomLanguageModel2:
    """
    Trigram Kneser-Ney
    """

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

        # calculate discount for Kneser-Ney
        # d ~ f1 / (f1 + 2f2)
        f1 = 0
        f2 = 0
        for count in bigramCounts.values():
            if count == 1:
                f1 += 1
            elif count == 2:
                f2 += 1
        d = f1 / (f1 + 2*f2)

        # calculate type(w,*) & type(*,w)
        type_bi1 = {}
        type_bi2 = {}
        for bigram in bigramCounts:
            type_bi1[bigram[0]] = type_bi1.get(bigram[0], 0) + 1
            type_bi2[bigram[1]] = type_bi2.get(bigram[1], 0) + 1

        # calculate type(w1,w2,*) & type(*,*,w3)
        type_tri1 = {}
        type_tri2 = {}
        for trigram in trigramCounts:
            type_tri1[(trigram[0], trigram[1])] = type_tri1.get((trigram[0], trigram[1]), 0) + 1
            type_tri2[trigram[2]] = type_tri2.get(trigram[2], 0) + 1

        # save word count and total for next test part
        self.unigram_count = unigramCounts
        self.bigram_count = bigramCounts
        self.trigram_count = trigramCounts
        self.type_bi1 = type_bi1  # type(w,*)
        self.type_bi2 = type_bi2  # type(*,w)
        self.type_bi_all = len(bigramCounts)  # type(*,*)
        self.type_tri1 = type_tri1  # type(w1,w2,*)
        self.type_tri2 = type_tri2  # type(*,*,w3)
        self.type_tri_all = len(trigramCounts)  # type(*,*,*)
        self.total = total  # total tokens
        self.d = d  # coefficient for discount

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here

        N = self.total  # total tokens
        d = self.d  # coefficient for discount
        d = 0.94  # alternative tuned parameter

        # start tag for trigram
        sentence = ['<r>'] + sentence + ['</r>']

        # total vocab number including UNK for laplace smoothing
        V = len(self.unigram_count.keys() | set(sentence))

        ### calculate probability ###
        """
        calculate P(w3|w1,w2)
        
        if w3 is UNK, use laplace unigram instead
        P(w3|w1,w2) = 1 / (N + V)
        
        if w3 is not UNK, and w2 is UNK
        P(w3|w1,w2) = 1 / V * Pcontinuation(w3)
        
        if w3, w2 are not UNK, and there is not bigram (w1, w2)
         - AD = max(c(w2,w3)-d,0) / c(w2)
         - λ(w2) = d / c(w2) * type(w2,*)
         - Pcontinuation(w3) = type(*,w3) / type(*,*)
        P(w3|w1,w2) = AD + λ(w2)Pcontinuation(w3) = p2
        
        if there is bigram (w1, w2)
         - AD = max(c(w1,w2,w3)-d,0) / c(w1,w2)
         - λ(w1,w2) = d / c(w1,w2) * type(w1,w2,*)
        P(w3|w1,w2) = AD + λ(w1,w2) * p2
        """
        # logP(W) = logP(<r>) + logP(<s>|<r>) + logP(w1|<r>,<s>) + logP(w2|<s>,w1) ...
        score = 0.0  # P(<r>) = P(<s>|<r>) = 1
        for i in range(2, len(sentence)):  # begin from the third index = logP(w1|<r>,<s>)
            w1 = sentence[i-2]
            w2 = sentence[i-1]
            w3 = sentence[i]
            if w3 in self.unigram_count:
                if w2 in self.unigram_count:
                    AD = max(self.bigram_count.get((w2, w3), 0) - d, 0) / self.unigram_count[w2]
                    lamb = d / self.unigram_count[w2] * self.type_bi1[w2]
                    Pcon = self.type_bi2[w3] / self.type_bi_all
                    p2 = AD + lamb * Pcon
                    if (w1, w2) in self.bigram_count:
                        AD = max(self.trigram_count.get((w1, w2, w3), 0) - d, 0) / self.bigram_count[(w1, w2)]
                        lamb = d / self.bigram_count[(w1, w2)] * self.type_tri1[(w1, w2)]
                        prob = AD + lamb * p2
                    else:
                        prob = p2
                else:
                    Pcon = self.type_bi2[w3] / self.type_bi_all
                    prob = (1 / V) * Pcon
            else:
                prob = 1 / (N + V)
            score += math.log(prob)
        return score