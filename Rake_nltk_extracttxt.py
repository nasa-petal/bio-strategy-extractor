#!/usr/bin/env python
# coding: utf-8

# In[10]:


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


if txt.count(". ") > 150:
    length = int(round(txt.count(". ")/10, 0))
else:
    length = 1


# In[4]:


# Remove punctuation and stopwords:
rmvp = [char for char in txt if char not in string.punctuation]
rmvp = ''.join(rmvp)
new_text =[word for word in rmvp.split() if word.lower() not in nltk.corpus.stopwords.words('english')]


# In[5]:


word_frequency = {}

for word in new_text:
    if word not in word_frequency:
        word_frequency[word] = 1
    else:
        word_frequency[word] = word_frequency[word] + 1


# In[6]:


maxfreq = max(word_frequency.values())
for word in word_frequency.keys():
    word_frequency[word] = (word_frequency[word]/maxfreq)


# In[7]:


slist = nltk.sent_tokenize(txt)
sscore = {}
for sent in slist:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequency.keys():
            if sent not in sscore.keys():
                sscore[sent] = word_frequency[word]
            else:
                sscore[sent] = sscore[sent] + word_frequency[word]


# In[12]:


summary = nlargest(length, sscore, key = sscore.get)
summ = ' '.join(summary)
print(summ, r.get_ranked_phrases())


# In[ ]:




