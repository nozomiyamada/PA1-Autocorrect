import math, collections

class UnigramLanguageModel:
  """Language model that uses unigram probabilities, ignoring unseen words."""

  def __init__(self, corpus):
    self.unigramProbs = {}
    self.train(corpus)

  def train(self, corpus):
    """Takes a HolbrookCorpus corpus, does whatever training is needed."""
    unigramCounts = {}
    total = 0
    for sentence in corpus.corpus:
      for datum in sentence.data:  
        token = datum.word
        unigramCounts[token] = unigramCounts.get(token, 0) + 1
        total += 1
    self.unigramProbs = {k: float(v) / total for k, v in unigramCounts.items()}
  
  def score(self, sentence):
    """Takes a list of strings, returns a score of that sentence."""
    score = 0.0 
    for token in sentence:
      # Ignore unseen words
      if token in self.unigramProbs:
        prob = self.unigramProbs[token]
        score += math.log(prob)
    return score
