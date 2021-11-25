from gensim.models import KeyedVectors

filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
mk_model = KeyedVectors.load_word2vec_format(filepath)   #DNA词向量长度为100