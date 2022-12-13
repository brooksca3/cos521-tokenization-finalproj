### tester on corpus of 2.5 million words, 
### reading the file and creating the dict took about 17 sec
### with 471604 resulting keys for n=8

import re
import random
import numpy as np

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return np.divide(e_x, e_x.sum(axis=0)) # only difference

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
  text = text.lower()
  # list of each space-separated word
  ls = text.split()
  dc = {}
  # consider 2 grams through n grams
  for word in ls:
    lim = min(len(word), n)
    slide(dc, word, lim, init_prob)
  return dc

def create_tok_arr(dc):
  ls = list(dc)
  tok_ls = []
  prob_ls = []
  for tok in ls:
    tok_ls.append(tok)
    prob_ls.append(dc[tok])
  return np.array(tok_ls), np.array(prob_ls)

# input a list of tokens, output a loss, right now just random, skewed towards words that end in vowels
def loss_proxy(tokens):
  vowel_end_counter = 0
  # vowel_ls = ['α', 'ε', 'υ', 'ο', 'ι', 'ω', 'η']
  vowel_ls = ['a', 'e', 'i', 'o', 'u']
  for tok in tokens:
    if tok[-1] in vowel_ls:
      vowel_end_counter += 1
  # boost the score for the words ending in vowels a bit
  score = 1 + 1 * random.random() - 5 * vowel_end_counter / len(tokens)
  return score
def normalize(probs):
  return probs / np.sum(probs)
# returns a list of tokens drawn from the dict according to the prob distribution
def draw_toks(probs, num_toks):
    return np.random.choice(len(probs), num_toks, p=normalize(probs), replace=False)

def iterate_probs(dc, n, batches, iters):
  tokens, probs = create_tok_arr(dc)
  num_to_exclude = int(n / (batches - 1))
  num_to_draw = int(num_to_exclude * batches)
  for _ in range(iters):
    if _ % 100 == 0:
      # r = np.sort(probs)
      # print(r[-10:])
      print(_)
      # print(sorted(probs, reverse=True))
    draws = draw_toks(probs, num_to_draw)
    scores = []
    for i in range(batches):
      excluded_indices = set(draws[i * num_to_exclude: (i + 1) * num_to_exclude])
      cur_idx = list(set(draws) - excluded_indices)
      scores.append(loss_proxy(tokens[cur_idx]))
    scores = softmax(scores)
    draw_prob = np.sum(probs[draws])
    for i in range(batches):
      indices = draws[i * num_to_exclude: (i + 1) * num_to_exclude]
      init_probability = np.sum(probs[indices])
      # multiplier = scores[i] * init_probability
      multiplier = 1 - (1 - scores[i]) * 0.35
      probs[indices] = np.multiply(probs[indices], multiplier)
    # probs[draws] = softmax(probs[draws]) * draw_prob
    probs[draws] /= np.sum(probs[draws])
    probs[draws] *= draw_prob
      # tokens = draws[i * num_to_exclude: (i + 1) * num_to_exclude]
      # for tok in draws[i * num_to_exclude: (i + 1) * num_to_exclude]:
      #   # if the loss was higher than avg when this token set was excluded
      #   cur_prob = dc[tok]
      #   if scores[i] > (1 / len(scores)):
      #     dc[tok] = 1 - ((1 - cur_prob) * (1 / len(scores)) * (1 / scores[i]))
      #   else:
      #     # if the loss was lower...
      #     dc[tok] = cur_prob * (scores[i]) * (1 / len(scores))
  return tokens, probs



def get_vowel_score(toks, probs):
  sprobs, stoks = (np.array(t) for t in zip(*sorted(zip(probs, toks), reverse=True)))
  # print(sprobs)
  vowel_end_counter = 0
  for i in range(100):
    print(sprobs[i], stoks[i], "\t")
    vowel_ls = ['a', 'e', 'i', 'o', 'u']
    if stoks[i][-1] in vowel_ls:
      vowel_end_counter += 1
  # x = (sorted(((v, k) for k, v in dc.items()), reverse=True))
  # vowel_end_counter = 0
  # print(x[:100])
  # for i in x[:100]:
  #   # vowel_ls = ['α', 'ε', 'υ', 'ο', 'ι', 'ω', 'η']
  #   vowel_ls = ['a', 'e', 'i', 'o', 'u']
  #   if i[1][-1] in vowel_ls:
  #     vowel_end_counter += 1
  return(vowel_end_counter)


with open('sherlock_holmes.txt', 'r') as f:
  text = f.read()
dc = prep_dict(text, 8, 0.5)
print(len(dc))


toks, probs = iterate_probs(dc, 36, 10, 10000)
print(probs)

print(get_vowel_score(toks, probs))