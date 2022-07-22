#!/usr/bin/env python
# coding: utf-8

# Steps for NLP Pipeline that we can implement in our algorithm after further literature research:
# 
# $\newline$1. Sentence segmentation: breaks the given paragraph into separate sentences.
# $\newline$2. Word tokenization: extract the words from each sentence one by one.
# $\newline$3. 'Parts of Speech' Prediction: identifying parts of speech.
# $\newline$4. Text Lemmatization: figure out the most basic form of each word in a sentence. "Germ" and "Germs" can have two different meanings and we should look to solve that.
# $\newline$5. 'Stop Words' Identification: English has a lot of filter words that appear very frequently and that introduces a lot of noise.
# $\newline$6. Dependency Parsing: uses the grammatical laws to figure out how the words relate to one another.
# $\newline$7. Entity Analysis: go through the text and identify all of the important words or “entities” in the text.
# $\newline$8. Pronouns Parsing: keeps track of the pronouns with respect to the context of the sentence.

# ## Step 1: sentence segmentation

# In[1]:


#pip install spacy
#spacy.cli.download("en_core_web_sm")


# In[2]:


import spacy


# In[3]:


nlp = spacy.load("en_core_web_sm")


# In[4]:


doc = nlp(u"While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications.")


# In[5]:


for sent in doc.sents:
    print(sent)


# ## Step 2: Word tokenization

# In[6]:


biomim = nlp(open('Abstract_textextraction.txt').read())
words_biomim = [word.text for word in biomim]
print(words_biomim)


# Nouns

# In[7]:


print("Noun phrases:", [chunk.text for chunk in biomim.noun_chunks])


# Verbs

# In[8]:


print("Verbs:", [token.lemma_ for token in biomim if token.pos_ == "VERB"])


# Named entities

# In[9]:


for entity in biomim.ents:
    print(entity.text, entity.label_)


# ## Step 3: Parts-of-speech prediction and Step 8: Pronouns Parsing

# In[10]:


for token in doc:
    # Print the token and its part-of-speech tag
    print(token.text, "-->", token.pos_)


# In[11]:


spacy.explain("PART")


# ## Step 4: Text Lemmatization

# In[12]:


#extract lemma for each token:
" ".join([token.lemma_ for token in doc])


# ## Step 5: 'Stop Words Identification'

# In[13]:


for token in doc:
    print(token.text,token.is_stop)


# In[14]:


# If we want to remove stop words:
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words

text = 'While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications.'
lst=[]
for token in text.split():
    if token.lower() not in stopwords:    #checking whether the word is not 
        lst.append(token)                    #present in the stopword list.
        
#Join items in the list
print("Original text  : ",text)
print("Text after removing stopwords  :   ",' '.join(lst))


# In[15]:


#Filtering stop words from text file:
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words

with open("Abstract_textextraction.txt") as f:
    text=f.read()
    
lst=[]
for token in text.split():
    if token.lower() not in stopwords:
        lst.append(token)

print('Original Text')        
print(text,'\n\n')

print('Text after removing stop words')
print(' '.join(lst))


# ## Step 6: Dependency Parsing

# In[16]:


for token in doc:
    print(token.text, "-->", token.dep_)


# ## Step 7: Entity Analysis

# In[17]:


for ent in doc.ents:
    print(ent.text, ent.label_)


# ## NN

# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
df = pd.read_json('golden.json')
df.head()


# In[19]:


df.isnull().sum()


# In[20]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.33, random_state=42)
print('Paper:', train['paper'].iloc[0])
print('Mesh Terms:', train['mesh_terms'].iloc[0])
print('Venue IDs:', train['venue_ids'].iloc[0])
print('Venue Names:', train['venue_names'].iloc[0])
print('Author Ids:', train['author_ids'].iloc[0])
print('Author Names:', train['author_names'].iloc[0])
print('Reference IDs:', train['reference_ids'].iloc[0])
print('Title:', train['title'].iloc[0])
print('Abstract:', train['abstract'].iloc[0])
print('Open Access:', train['isOpenAccess'].iloc[0])
print('Full Doc Link:', train['fullDocLink'].iloc[0])
print('PeTaL ID:', train['petalID'].iloc[0])
print('doi:', train['doi'].iloc[0])
print('Level 1:', train['level1'].iloc[0])
print('Level 2:', train['level2'].iloc[0])
print('Level 3:', train['level3'].iloc[0])
print('Biomimicry:', train['isBiomimicry'].iloc[0])
print('url:', train['url'].iloc[0])
print('Mag Terms:', train['mag_terms'].iloc[0])
print('Species:', train['species'].iloc[0])
print('Absolute Relevancy:', train['absolute_relevancy'].iloc[0])
print('Relative Relevancy:', train['relative_relevancy'].iloc[0])
print('Training Data Shape:', train.shape)
print('Testing Data Shape:', test.shape)


# In[21]:


import spacy
nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)
INFO_text = [text for text in train[train['mesh_terms'] == 'Osteomalacia']['paper']]
IS_text = [text for text in train[train['mesh_terms'] == 'Bone and Bones']['paper']]
INFO_clean = cleanup_text(INFO_text)
INFO_clean = ' '.join(INFO_clean).split()
IS_clean = cleanup_text(IS_text)
IS_clean = ' '.join(IS_clean).split()
INFO_counts = Counter(INFO_clean)
IS_counts = Counter(IS_clean)
INFO_common_words = [word[0] for word in INFO_counts.most_common(20)]
INFO_common_counts = [word[1] for word in INFO_counts.most_common(20)]


# In[22]:


IS_common_words = [word[0] for word in IS_counts.most_common(20)]
IS_common_counts = [word[1] for word in IS_counts.most_common(20)]


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
#from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction import _stop_words
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
import string
import re
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en import English
parser = English()


# In[24]:


STOPLIST = set(stopwords.words('english'))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
def get_params(self, deep=True):
        return {}
    
def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text
def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


# In[25]:


def printNMostInformative(vectorizer, clf, N):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
clf = LinearSVC()

pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
# data
train1 = train['paper'].tolist()
labelsTrain1 = train['mesh_terms'].tolist()
test1 = test['paper'].tolist()
labelsTest1 = test['mesh_terms'].tolist()
# train
pipe.fit(train1, labelsTrain1)

####
# test
preds = pipe.predict(test1)

####
print("accuracy:", accuracy_score(labelsTest1, preds))
print("Top 10 features used to predict: ")

printNMostInformative(vectorizer, clf, 10)
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
transform = MultiLabelBinarizer().fit_transform(train1, labelsTrain1)
vocab = vectorizer.get_feature_names()
for i in range(len(train1)):
    s = ""
    indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
    numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
    for idx, num in zip(indexIntoVocab, numOccurences):
        s += str((vocab[idx], num))


# In[ ]:


from sklearn import metrics
print(metrics.classification_report(labelsTest1, preds, 
                                    target_names=df['mesh_terms'].unique()))


# In[ ]:




