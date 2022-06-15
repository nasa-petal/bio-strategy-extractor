import gensim.downloader as api
import numpy as np

class W2VTextSimilarity:
    def __init__(self, text, target, w2v_model):
        self.txt = text
        self.target = target
        self.sentences = txt.split('.')
        self.target_words = target.split()
        self.model = w2v_model
    
    def compute_similarity(self):
        avg_target_vec = None
        target_sentence_mat = []
        for word in self.target_words:
            if word in model.key_to_index:
                target_sentence_mat.append(model[word])
        target_sentence_mat = np.array(target_sentence_mat)
        avg_target_vec = np.mean(target_sentence_mat, axis=0)

        scores_dict = {}
        scores = []
        for sentence in self.sentences:
            words = sentence.split()
            if words:
                sentence_mat = []
                for word in words:
                    if word in model.key_to_index:
                        sentence_mat.append(model[word])
                sentence_mat = np.array(sentence_mat)
                # print(sentence_mat.shape)
                avg_vec = np.mean(sentence_mat,axis=0) 
                cos_sim = np.dot(avg_vec, avg_target_vec)/(np.linalg.norm(avg_vec)*np.linalg.norm(avg_target_vec))
                scores_dict[cos_sim] = sentence
                scores.append(cos_sim)
        return scores_dict, scores

txt = "While scanning the water for these hydrodynamic signals at a swimming speed in the order of meters per second, the seal keeps its long and flexible whiskers in an abducted position, largely perpendicular to the swimming direction. Remarkably, the whiskers of harbor seals possess a specialized undulated surface structure, the function of which was, up to now, unknown. Here, we show that this structure effectively changes the vortex street behind the whiskers and reduces the vibrations that would otherwise be induced by the shedding of vortices from the whiskers (vortex-induced vibrations). Using force measurements, flow measurements and numerical simulations, we find that the dynamic forces on harbor seal whiskers are, by at least an order of magnitude, lower than those on sea lion (Zalophus californianus) whiskers, which do not share the undulated structure. The results are discussed in the light of pinniped sensory biology and potential biomimetic applications."
target = "A small diameter fiber with an undulated surface structure reduces vibrations caused by drag forces" 
model = api.load("word2vec-google-news-300")

sim = W2VTextSimilarity(txt, target, model)
mapping, scores = sim.compute_similarity()
scores.sort()
for score in scores:
    print(str(score) + " " + mapping[score])

