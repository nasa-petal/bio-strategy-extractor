#!/usr/bin/env python
# coding: utf-8

# In[17]:


import nltk
import string
from heapq import nlargest
from rake_nltk import Rake
import numpy as np

r = Rake()

# Note: Can use a website or directory with the text file in place of this line:
#txt = "While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications."
#txt = "This study shows that although annual fluctuations occur in the main prey types of Skomer Puffins, food availability does not appear to limit breeding success. Most birds found food close to the colony, showing peaks of activity early in the day and again in late afternoon; and two out of four experimental pairs were able to rear 'twins', though the growth-rates of these were less than for single chicks. Some interesting observations are given on the feeding of the chick and on kleptoparasitism of adults by Jackdaws and gulls."
txt = "Silks are strong protein fibers produced by a broad array of spiders and insects. The vast majority of known silks are large, repetitive proteins assembled into extended β-sheet structures. Honeybees, however, have found a radically different evolutionary solution to the need for a building material. The 4 fibrous proteins of honeybee silk are small (∼30 kDa each) and nonrepetitive and adopt a coiled coil structure. We examined silks from the 3 superfamilies of the Aculeata (Hymenoptera: Apocrita) by infrared spectroscopy and found coiled coil structure in bees (Apoidea) and in ants (Vespoidea) but not in parasitic wasps of the Chrysidoidea. We subsequently identified and sequenced the silk genes of bumblebees, bulldog ants, and weaver ants and compared these with honeybee silk genes. Each species produced orthologues of the 4 small fibroin proteins identified in honeybee silk. Each fibroin contained a continuous predicted coiled coil region of around 210 residues, flanked by 23–160 residue length N- and C-termini. The cores of the coiled coils were unusually rich in alanine. There was extensive sequence divergence among the bee and ant silk genes (<50% similarity between the alignable regions of bee and ant sequences), consistent with constant and equivalent divergence since the bee/ant split (estimated to be 155 Myr). Despite a high background level of sequence diversity, we have identified conserved design elements that we propose are essential to the assembly and function of coiled coil silks."
r.extract_keywords_from_text(txt)

#r.get_ranked_phrases()[0:10]
r.get_ranked_phrases()


# In[18]:


r.get_ranked_phrases_with_scores()


# In[19]:


# Text summarization method 1


# In[20]:


if txt.count(". ") > 150:
    length = int(round(txt.count(". ")/10, 0))
else:
    length = 1


# In[21]:


# Remove punctuation and stopwords:
rmvp = [char for char in txt if char not in string.punctuation]
rmvp = ''.join(rmvp)
new_text =[word for word in rmvp.split() if word.lower() not in nltk.corpus.stopwords.words('english')]


# In[22]:


word_frequency = {}

for word in new_text:
    if word not in word_frequency:
        word_frequency[word] = 1
    else:
        word_frequency[word] = word_frequency[word] + 1


# In[23]:


maxfreq = max(word_frequency.values())
for word in word_frequency.keys():
    word_frequency[word] = (word_frequency[word]/maxfreq)


# In[24]:


slist = nltk.sent_tokenize(txt)
sscore = {}
for sent in slist:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequency.keys():
            if sent not in sscore.keys():
                sscore[sent] = word_frequency[word]
            else:
                sscore[sent] = sscore[sent] + word_frequency[word]


# In[25]:


summary = nlargest(length, sscore, key = sscore.get)
summ = ' '.join(summary)
print(summ, r.get_ranked_phrases())


# Text Summarization that incorporates what Rake does.

# In[26]:


from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


# In[27]:


def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = ["Silks are strong protein fibers produced by a broad array of spiders and insects. The vast majority of known silks are large, repetitive proteins assembled into extended β-sheet structures. Honeybees, however, have found a radically different evolutionary solution to the need for a building material. The 4 fibrous proteins of honeybee silk are small (∼30 kDa each) and nonrepetitive and adopt a coiled coil structure. We examined silks from the 3 superfamilies of the Aculeata (Hymenoptera: Apocrita) by infrared spectroscopy and found coiled coil structure in bees (Apoidea) and in ants (Vespoidea) but not in parasitic wasps of the Chrysidoidea. We subsequently identified and sequenced the silk genes of bumblebees, bulldog ants, and weaver ants and compared these with honeybee silk genes. Each species produced orthologues of the 4 small fibroin proteins identified in honeybee silk. Each fibroin contained a continuous predicted coiled coil region of around 210 residues, flanked by 23–160 residue length N- and C-termini. The cores of the coiled coils were unusually rich in alanine. There was extensive sequence divergence among the bee and ant silk genes (<50% similarity between the alignable regions of bee and ant sequences), consistent with constant and equivalent divergence since the bee/ant split (estimated to be 155 Myr). Despite a high background level of sequence diversity, we have identified conserved design elements that we propose are essential to the assembly and function of coiled coil silks."]
    #sentences = ["This study shows that although annual fluctuations occur in the main prey types of Skomer Puffins, food availability does not appear to limit breeding success. Most birds found food close to the colony, showing peaks of activity early in the day and again in late afternoon; and two out of four experimental pairs were able to rear 'twins', though the growth-rates of these were less than for single chicks. Some interesting observations are given on the feeding of the chick and on kleptoparasitism of adults by Jackdaws and gulls."]
    #sentences = ["While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications."]
    
    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences


# In[28]:


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)


# In[29]:


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


# In[30]:


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))

# let's begin
generate_summary("Conservation_of_Essential_Design_Features_In_Coiled_Coil_Silks.txt", 2)


# In[31]:


print(summ, generate_summary( "Conservation_of_Essential_Design_Features_In_Coiled_Coil_Silks.txt", 2))


# In[16]:


# try on other abstracts (petalai.org)
# See above for this.

