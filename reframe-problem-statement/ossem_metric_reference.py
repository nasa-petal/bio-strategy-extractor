# The OSSEM metric computes the Originality of Syntactictics and SEMantics. It uses a combination of Rouge and TF-IDF cosine similarity
# to compare a generated text to a seed text to make sure that the word choice and word meaning of the generated text was different enough
# from the seed text. All the functions and procedures necessary to compute this metric on your own data are included in this program.

# OSSem (Originality of Syntactics and Semantics)
# 0.0 is maximal originality, 1.0 is minimal, trying to minimize in this use case
# input: each reframed sentence, and the original problem statement
# output: originality score of each reframed sentence
# pipeline: rouge-l f1-score
#           tf-idf of all sentences
#           cosine sim
#           output

# ! pip install rouge
# ! pip install torch

import torch

from rouge import Rouge 

def extract_vocab(documents): # list of list of strings: documents
  vocab = []
  for document in documents:
    for token in document:
      if token not in vocab:
        vocab.append(token)
  return vocab # unique words in a set of documents

def tf(frequencies, document_index, token): # list of dictionaries {string : int}: frequencies || int: document_index || string: token
  document_word_count = 0
  for frequency in frequencies[document_index].items():
    document_word_count += frequency[1]
  if document_word_count == 0:
    return 0
  return 1.0 * frequencies[document_index][token] / document_word_count # term frequency in a single document (weighted to make frequencies across all document sizes on the same scale)

def idf(frequencies, token): # list of dictionaries {string : int}: frequencies || string: token
  documents_in_total = len(frequencies)
  documents_with_token = 0
  for document in frequencies:
    if document[token] != 0:
      documents_with_token += 1
  return math.log(1.0 * documents_in_total / documents_with_token) # term rarity

def tf_idf(documents): # list of list of strings: documents
  # set up the vocab for future reference
  vocab = extract_vocab(documents)
  # set up the frequencies for future reference
  frequencies = []
  document_frequencies = {}
  for term in vocab:
    document_frequencies[term] = 0
  for document in documents:
    document_frequencies = {}
    for term in vocab:
      document_frequencies[term] = 0
    for token in document:
      document_frequencies[token] += 1
    frequencies.append(document_frequencies)
  # generating the embeddings for each document
  documents_embeddings = []
  for i in range(len(documents)):
    document_embeddings = []
    for token in documents[i]:
      document_embeddings.append(tf(frequencies, i, token) * idf(frequencies, token))
    documents_embeddings.append(document_embeddings)
  return documents_embeddings # each embedding is in the order of the documents list passed in the arguments

def cosine_similarity(vec_1, vec_2):
  vec_1 = np.array(vec_1)
  vec_2 = np.array(vec_2)
  numerator = np.dot(vec_1, vec_2)
  denominator = np.sqrt(np.dot(vec_1, vec_1)) * np.sqrt(np.dot(vec_2, vec_2))
  return numerator / denominator

def extract_n_grams(document, n_size): # list of strings: document || int: n_size
  n_grams = []
  for i in range(len(document)):
    n_gram = []
    for j in range(n_size):
      n_gram.append(document[j])
    n_grams.append(n_gram)
  return n_grams

def rouge_l_f1(comp_produced, human_produced):
  return Rouge().get_scores(comp_produced, human_produced)[0]['rouge-l']['f']

# use a variation of this code to test this metric:
# 
# problems_in_order = []
# for problem in problem_statements:
#   problem_1 = [problem[0].split(" ")]
#   for reframed in problem[1]:
#     reframed = reframed.split(" ")
#     problem_1.append(reframed)
#   problems_in_order.append(problem_1)
# for problem in problems_in_order:
#   documents = problem
#   human_produced = " ".join(documents[0])
#   comp_produced = " ".join(documents[1])
#   embeddings = tf_idf(documents)
#   human_produced_embed = embeddings[0]
#   comp_produced_embed = embeddings[1]
#   cos_sim = cosine_similarity(human_produced_embed, comp_produced_embed)
#   rouge = rouge_l_f1(human_produced, comp_produced)
#   ossem = torch.sigmoid(torch.tensor(math.log(rouge + cos_sim)))
#   problem.append(ossem.item())
# problems_in_order
