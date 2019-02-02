import math
class CustomLanguageModel:
    """
    Kneser-Ney smoothing
    """

    def __init__(self, corpus):
        self.train(corpus)
        self.i = 0

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

        # calculate d for Kneser-Ney
        # d ~ f1 / (f1 + 2f2)
        f1 = 0
        f2 = 0
        for count in bigramCounts.values():
            if count == 1:
                f1 += 1
            elif count == 2:
                f2 += 1
        d = f1 / (f1 + 2*f2)

        # save word count and total for next test part
        self.unigram_count = unigramCounts
        self.bigram_count = bigramCounts
        self.total = total
        self.d = d

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
        """
        # TODO your code here

        # initialize count with trained data
        unigram_count = self.unigram_count.copy()
        bigram_count = self.bigram_count.copy()
        N = self.total
        d = self.d  # coefficient

        ### calculate probability ###
        """
        P(wi|wi-1) = AD + λ(wi-1)Pcontinuation(wi)
        AD = max(c(wi-1,wi)-d,0) / c(wi-1)
        λ(wi-1) = d / c(wi-1) * |{w:c(wi-1,w)}>0|
        Pcontinuation(wi) = type(:,wi) / type(:,:)
        """
        # logP(W) = logP(<s>) + logP(w1|<s>) + logP(w2|w1) + logP(w3|w2) ...
        score = 0.0  # P(<s>) = 1
        type_all = len(bigram_count)  # type(:,:)

        for i in range(1, len(sentence)):  # begin from the second index = logP(w1|<s>)
            w1 = sentence[i-1]
            w2 = sentence[i]

            if (w1, w2) in bigram_count:  # neither wi nor wi-1 is UNK
                AD = max(bigram_count[(w1, w2)]-d, 0) / unigram_count[w1]
                lamb = d / unigram_count[w1]
                type_wi = 0
                for bigram in bigram_count:
                    if bigram[1] == w2:
                        type_wi += 1
                Pcon = type_wi / type_all
                prob = AD + lamb * Pcon
            else:
                prob = 1 / N
            score += math.log(prob)

        if self.i % 1000 == 0:
            print(self.i)
        self.i += 1
        return score