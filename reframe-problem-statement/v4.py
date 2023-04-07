# this model implements the multimodal chain-of-thought architecture, first described by https://arxiv.org/pdf/2302.00923.pdf
# ideally, the model should be trained by a multimodal dataset, with intermediate "thinking" steps, as shown by ScienceQA and others

import gensim.downloader
import spacy
import torch
import torchvision.transforms as T
import math

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from PIL import Image
from torch import nn
from torchvision.models import resnet101

class ProjectionMatrix(nn.Module):
  def __init__(self):
    super(ProjectionMatrix, self).__init__()
    self.linear = nn.Linear(25 * 25, 300)

  def forward(self, x):
    x = x.view(-1, 25 * 25)
    x = self.linear(x)
    x = x.view(-1, 300, 1)
    return x

def main():
  # set up external encoding models
  torch.set_grad_enabled(False)
  sp = spacy.load('en_core_web_sm')
  glove_vector = gensim.downloader.load('fasttext-wiki-news-subwords-300')
  detr_model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained = True)
  detr_model.eval()
  softmax_fn = nn.Softmax(dim=1)
  # set up data
  sample_x_lang = "hello there why would you be a goose what even was going through your tiny mind"
  strawberries = Image.open("/content/drive/MyDrive/strawberries_for_detection.jpg")
  elephants = Image.open("/content/drive/MyDrive/elephants_for_detection.jpg")
  sample_x_vis = [strawberries, elephants]
  # forward step
  H_lang = encoding_lang(sample_x_lang)
  H_vis = encoding_vis(sample_x_vis)
  H_fuse = interaction(H_lang, H_vis)
  rationale = decoder(H_fuse)
  H_lang += rationale
  H_lang = encoding_lang(H_lang)
  H_fuse = interaction(H_lang, H_vis)
  answer = decoder(H_fuse)

def encoding_lang(x_lang):
  x_lang = sp(x_lang)
  H_lang = []
  for token in x_lang:
    H_lang.append(glove_vector.wv[token.text])
  return H_lang

def encoding_vis(x_vis):
  transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  patch_level_features = []
  for image in x_vis:
    image = transform(image).unsqueeze(0)
    outputs = detr_model(image)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [
      detr_model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),
      detr_model.transformer.encoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: enc_attn_weights.append(output[1])),
      detr_model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_attn_weights.append(output[1])),
    ]
    outputs = detr_model(image)
    for feature in conv_features:
      patch_level_features.append(feature["0"].tensors[0])
    for hook in hooks:
      hook.remove()
  patch_level_features = torch.stack(patch_level_features, dim=0)
  W_h = ProjectionMatrix()
  H_vis = W_h(patch_level_features)
  return H_vis

def interaction(H_lang, H_vis):
  Q = H_lang
  K = H_vis
  V = H_vis
  softmax_input = torch.div(torch.matmul(Q, torch.transpose(K, 0, 1)), math.sqrt(300))
  softmax_output = softmax_fn(softmax_input)
  H_attn_vision = torch.matmul(softmax_output, V)
  W_l = ProjectionMatrix()
  W_v = ProjectionMatrix()
  lamb = torch.sigmoid(W_l(H_lang) + W_v(H_attn_vision))
  H_fuse = (1 - lamb) * H_lang + lamb * H_attn_vision
  return H_fuse

def decoder(H_fuse):
  predictions = []
  for token_vec in H_fuse:
    predictions.append(glove_vector.most_similar(positive=[token_vec], topn=1))
  return predictions

if __name__ == "__main__":
    main()
