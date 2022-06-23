
import numpy as np

class W2VTextSimilarity:
    def __init__(self, text, target, w2v_model):
        self.txt = text
        self.target = target
        self.sentences = text.split('.')
        self.target_words = target.split()
        self.model = w2v_model
    
    def compute_similarity(self):
        avg_target_vec = None
        target_sentence_mat = []
        for word in self.target_words:
            if word in self.model.key_to_index:
                target_sentence_mat.append(self.model[word])
        target_sentence_mat = np.array(target_sentence_mat)
        avg_target_vec = np.mean(target_sentence_mat, axis=0)

        scores_dict = {}
        scores = []
        for sentence in self.sentences:
            words = sentence.split()
            if words:
                sentence_mat = []
                for word in words:
                    if word in self.model.key_to_index:
                        sentence_mat.append(self.model[word])
                sentence_mat = np.array(sentence_mat)
                # print(sentence_mat.shape)
                avg_vec = np.mean(sentence_mat,axis=0) 
                cos_sim = np.dot(avg_vec, avg_target_vec)/(np.linalg.norm(avg_vec)*np.linalg.norm(avg_target_vec))
                scores_dict[cos_sim] = sentence
                scores.append(cos_sim)
        return scores_dict, scores