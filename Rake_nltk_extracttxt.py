#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import string
from heapq import nlargest
from rake_nltk import Rake
import numpy as np

r = Rake()

# Note: Can use a website or directory with the text file in place of this line:
txt = "While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications."

r.extract_keywords_from_text(txt)

#r.get_ranked_phrases()[0:10]
r.get_ranked_phrases()


# In[2]:


r.get_ranked_phrases_with_scores()


# In[3]:


# Text summarization method 1


# In[4]:


if txt.count(". ") > 150:
    length = int(round(txt.count(". ")/10, 0))
else:
    length = 1


# In[5]:


# Remove punctuation and stopwords:
rmvp = [char for char in txt if char not in string.punctuation]
rmvp = ''.join(rmvp)
new_text =[word for word in rmvp.split() if word.lower() not in nltk.corpus.stopwords.words('english')]


# In[6]:


word_frequency = {}

for word in new_text:
    if word not in word_frequency:
        word_frequency[word] = 1
    else:
        word_frequency[word] = word_frequency[word] + 1


# In[7]:


maxfreq = max(word_frequency.values())
for word in word_frequency.keys():
    word_frequency[word] = (word_frequency[word]/maxfreq)


# In[8]:


slist = nltk.sent_tokenize(txt)
sscore = {}
for sent in slist:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequency.keys():
            if sent not in sscore.keys():
                sscore[sent] = word_frequency[word]
            else:
                sscore[sent] = sscore[sent] + word_frequency[word]


# In[9]:


summary = nlargest(length, sscore, key = sscore.get)
summ = ' '.join(summary)
print(summ, r.get_ranked_phrases())


# Text Summarization that incorporates what Rake does.
# Source: https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70

# In[10]:


from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


# In[11]:


def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = ["While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications."]

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences


# In[19]:


def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: 
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


# In[20]:


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences =  read_article(file_name)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)        

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summarize Text: \n", ". ".join(summarize_text))

generate_summary("Abstract_textextraction.txt", 2)


# In[21]:


print(summ, generate_summary( "Abstract_textextraction.txt", 2))


# In[ ]:




