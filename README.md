# PA1-Autocorrect

training data
* sentence: 660
* word type: 1661
* bigram type: 6850
* tokens: 11250

most frequent unigram
('the', 771), ('<s>', 659), ('and', 466), ('to', 314), ('a', 287), ('i', 281), ('you', 169), ('in', 150), ('of', 142), ('they', 121), ('on', 120), ('was', 116), ('he', 113), ('went', 113), ('is', 106), ('it', 106), ('my', 101), ('said', 90), ('when', 72), ('for', 68)
  
most frequent bigram
(('<s>', 'the'), 94), (('and', 'the'), 73), (('to', 'the'), 71), (('in', 'the'), 61), (('<s>', 'i'), 60), (('of', 'the'), 49), (('on', 'the'), 45), (('<s>', 'they'), 36), (('went', 'to'), 35), (('<s>', 'when'), 29), (('i', 'am'), 27), (('<s>', 'my'), 22), (('the', 'police'), 22), (('and', 'i'), 22), (('the', 'old'), 22), (('the', 'man'), 21), (('the', 'end'), 21), (('and', 'he'), 21), (('<s>', 'one'), 20), (('end', '</s>'), 20)

| |correct|total | accuracy|
|:-:|--:|--:|--:|
|Unsmoothed Unigram LM |6 |471 |0.012739 |
|Uniform LM|26 |471 |0.055202 |
|Laplace Unigram LM |52 | 471| 0.110403 |
|Laplace Bigram LM|69 | 471| 0.146497 |
|Laplace Trigram LM|81 | 471| 0.171975 |
|Laplace 4gram LM|75 | 471| 0.159236 |
|Stupid Backoff Bigram LM k=0.4|85 | 471|0.180467 |
|Stupid Backoff Bigram LM k=0.6|83 | 471|0.176221 |
|Stupid Backoff Bigram LM k=0.8|83 | 471|0.176221 |
|Stupid Backoff Trigram LM k=0.4|90 | 471|0.191083 |
|Stupid Backoff Trigram LM k=0.6|91 | 471|0.193206 |
|Stupid Backoff Trigram LM k=0.8|90 | 471|0.191083 |
|Bigram Kneser-Ney LM d=0.40|115 | 471|0.244161 |
|Bigram Kneser-Ney LM d=0.60|120 | 471|0.254777 |
|Bigram Kneser-Ney LM d=0.77|123 | 471|0.261146 |
|Bigram Kneser-Ney LM d=0.80|126 | 471|0.267516 |
|Bigram Kneser-Ney LM d=0.90|119 | 471|0.252654 |
|Trigram Kneser-Ney LM d=0.50|116 | 471|0.246285 |
|Trigram Kneser-Ney LM d=0.77|122 | 471|0.259023 |
|Trigram Kneser-Ney LM d=0.92|124 | 471|0.263270 |
|Trigram Kneser-Ney LM d=0.94|126 | 471|0.267516 |
|Trigram Kneser-Ney LM d=0.95|124 | 471|0.263270 |
