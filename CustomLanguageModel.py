import math
class CustomLanguageModel:
    """
    Kneser-Ney smoothing
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
        total = 0
        for sentence in corpus.corpus:
            for i in range(len(sentence.data) - 1):  # -1 for the last bigram pair
                w1 = sentence.data[i].word
                w2 = sentence.data[i+1].word
                unigramCounts[w1] = unigramCounts.get(w1, 0) + 1
                bigramCounts[(w1, w2)] = bigramCounts.get((w1, w2), 0) + 1  # key is tuple (w1, w2)
                total += 1
            # for the last index (count </s> for P(</s>))
            last_token = sentence.data[-1].word
            unigramCounts[last_token] = unigramCounts.get(last_token, 0) + 1

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
        type_num1 = {}
        type_num2 = {}
        for bigram in bigramCounts:
            type_num1[bigram[0]] = type_num1.get(bigram[0], 0) + 1
            type_num2[bigram[1]] = type_num2.get(bigram[1], 0) + 1

        # save word count and total for next test part
        self.unigram_count = unigramCounts
        self.bigram_count = bigramCounts
        self.type_num1 = type_num1  # type(w,*)
        self.type_num2 = type_num2  # type(*,w)
        self.type_all = len(bigramCounts)  # type(*,*)
        self.total = total  # total tokens
        self.d = d  # coefficient for discount

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here

        N = self.total  # total tokens
        d = self.d  # coefficient for discount

        # total vocab number including UNK for laplace smoothing
        V = len(self.unigram_count.keys() | set(sentence))

        ### calculate probability ###
        """
        if wi-1, wi is not UNK
         - AD = max(c(wi-1,wi)-d,0) / c(wi-1)
         - λ(wi-1) = d / c(wi-1) * type(wi-1,*)
         - Pcontinuation(wi) = type(*,wi) / type(*,*)
        P(wi|wi-1) = AD + λ(wi-1)Pcontinuation(wi)
        
        if wi-1 = UNK, wi is not UNK
        assume that λ = 1 / V(all vocab number)
        P(wi|wi-1) = λ * Pcontinuation(wi)
        
        if wi is UNK, use laplace unigram instead
        P(wi|wi-1) = 1 / (N + V)
        """
        # logP(W) = logP(<s>) + logP(w1|<s>) + logP(w2|w1) + logP(w3|w2) ...
        score = 0.0  # P(<s>) = 1
        for i in range(1, len(sentence)):  # begin from the second index = logP(w1|<s>)
            w1 = sentence[i-1]
            w2 = sentence[i]
            if w1 in self.unigram_count and w2 in self.unigram_count:
                AD = max(self.bigram_count.get((w1, w2), 0) - d, 0) / self.unigram_count[w1]
                lamb = d / self.unigram_count[w1] * self.type_num1[w1]
                Pcon = self.type_num2[w2] / self.type_all
                prob = AD + lamb * Pcon
            elif w2 in self.unigram_count:  # w1 = UNK, w2 != UNK
                lamb = 1 / V
                Pcon = self.type_num2[w2] / self.type_all
                prob = lamb * Pcon
            else:
                prob = 1 / (N + V)  # laplace unigram
            score += math.log(prob)
        return score