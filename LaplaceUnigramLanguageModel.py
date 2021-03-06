import math
class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
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
    unigramCounts = {}
    total = 0
    for sentence in corpus.corpus:
      for datum in sentence.data:
        token = datum.word
        unigramCounts[token] = unigramCounts.get(token, 0) + 1
        total += 1  # token number

    # save word count and total for add-one in the next test part
    self.count = unigramCounts
    self.total = total

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here

    # initialize count with trained data
    unigram_count = self.count.copy()
    N = self.total

    # make a new key for UNK, add-one later
    for token in sentence:
      if token not in unigram_count:
        unigram_count[token] = 0

    # calcutate lopP(<s>) + logP(w1) + logP(w2) + ...
    score = 0.0  # P(<s>) = 1
    V = len(unigram_count)  # the number of vocab including UNK
    for word in sentence:
      prob = float((unigram_count[word] + 1) / (N + V))  # c(w) + 1 / N + V
      score += math.log(prob)

    return score
