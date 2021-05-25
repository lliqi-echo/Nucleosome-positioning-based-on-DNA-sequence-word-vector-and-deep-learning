import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors 
import csv

seqDict1={} 
f  = open('data/data28099/human n-i.txt','r')
i = 0
for lines in f:
    ls = lines.strip('\n')
    name = "H.sapiens" + "." + "nucleosome.inhibiting" + "." +str(i)
    seqDict1[name] = ls[1:]
    i +=1

seqDict2={} 
f  = open('data/data28099/human n-f.txt','r')
i = 0
for lines in f:
    ls = lines.strip('\n')
    name = "H.sapiens" + "." + "nucleosome.forming" + "." +str(i)
    seqDict2[name] = ls[1:]
    i +=1

seqDict = dict(seqDict1, **seqDict2) 

def getKmers(sequence, size):
    sentence=[]
    for x in range(len(sequence) - size + 1):
        word = sequence[x:x+size]
        sentence.append(word)
    return sentence  


def data_preprocess(seqdict, k):
    corpus=[]
    num_words=[]
    for key,value in seqdict.items():
        seq = getKmers(seqdict[key],k)
        corpus.append(seq)
        for word in seq:
            num_words.append(word)
    num_of_words=len(num_words)
    return corpus, num_of_words
k=4 #k-mer=3-6
corpus, num_of_words1=data_preprocess(seqDict, k)

model = Word2Vec(corpus,min_count=1,size=100,sg=1,hs=1,window=5)
model.wv.save("word2vec/model-dnavec.wv")


d2v_file = open('new-path_to_save','w')
csv_writer = csv.writer(d2v_file)
i =0
for key,value in d2v.items():
    vector = value
    vector = list(vector)  
    label = key.split('_')[0]
    if label == 'nucleosomal':
        a = 1
    else: 
        a = 0
    vector.append(a)
    csv_writer.writerow(vector)
    i +=1
d2v_file.close()