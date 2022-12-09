### tester on corpus of 2.5 million words, 
### reading the file and creating the dict took about 17 sec
### with 471604 resulting keys 

import re
# helper for prep_dict
def slide(dc, word, lim, init_prob):
  for i in range(len(word) - 1):
    for j in range(2, lim + 1):
      if i + j > len(word):
        break
      cur_sub = word[i : i + j]
      dc[cur_sub] = init_prob

# text = corpus to build dict from
# n = highest n-gram wanted in dict (e.g., substrings up to length n)
# init_prob = initial probability for each potential token
# we assume that every 1-gram is already in the tokenizer; i.e. every element of set(text) is in the tokenizer
def prep_dict(text, n, init_prob):
  # replace all whitespace with a single space
  text = re.sub(r'\s+', ' ', text)
  # get rid of non-alphabetic characters (punctuation / other stuff will be their own tokens)
  # might need to change this if the given language has different characters / script
  # text = re.sub(r'[^α-ωΑ-Ω ]', '', text)
  text = re.sub(r'[^a-zA-Z ]', '', text)
  # list of each space-separated word
  ls = text.split()
  dc = {}
  # consider 2 grams through n grams
  for word in ls:
    lim = min(len(word), n)
    slide(dc, word, lim, init_prob)
  return dc
