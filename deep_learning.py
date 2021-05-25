from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os 
import shutil
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp
from gensim.models import Word2Vec
from gensim.models import KeyedVectors 

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

wv = KeyedVectors.load("word2vec/model-dnavec.wv", mmap='r')

def getKmers(sequence, size):
    sentence=[]
    for x in range(len(sequence) - size + 1):
        word = sequence[x:x+size]
        sentence.append(word)
    return sentence  

d2v = dict()
k=4 #k-mer
for key,value in seqDict.items():
    seq = getKmers(seqDict[key],k)  
    total = 0
    for word in seq:
        total= total + wv[word]
    vector= total/len(seq)   
    d2v[key]=vector

vector_shape=100

def seqdata_reader(file_list, data_dict):
    q, p=file_list.shape
    data_count = q
    label_set = set()
    for i in range(q):
        data_key, label = file_list[i]
        label_set.add(label)
        class_dim = len(label_set)
    print("class dim:{0} data count:{1}".format(class_dim, data_count))

    def reader():
        for i in range(q):
            data_key, label = file_list[i]
            seq = getKmers(data_dict[data_key],k)
            seq_numpy = np.zeros(shape=(147-k+1, vector_shape)) 
            for a in range(len(seq)):  
                seq_numpy[a]=wv[seq[a]]
              
            yield seq_numpy, int(label)

    return reader

def CNN(data):
    conv_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=(3,100),
        num_filters=64,
        pool_size=2,
        pool_stride=2,
        act="tanh")    

    conv_2 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=(4,100),
        num_filters=64,
        pool_size=2,
        pool_stride=2,
        act="tanh")

    conv_3 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=(5,100),
        num_filters=64,
        pool_size=2,
        pool_stride=2,
        act="tanh")


    fc = fluid.layers.fc(input=[conv_1, conv_2, conv_3], size=144, act='relu')
    
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)  

    prediction = fluid.layers.fc(input=drop, size=2, act="softmax") 
       
    return prediction


def gru_lstm(data):
    r_data = fluid.layers.reshape(x=data, shape=[-1,147-k+1, vector_shape])
    position_tensor = fluid.layers.add_position_encoding(input=r_data, alpha=1.0, beta=1.0)

    cell1 = fluid.layers.GRUCell(hidden_size=vector_shape, activation=fluid.layers.relu)
    fw1, _ = fluid.layers.rnn(cell=cell1, inputs=r_data)
    bw1, _ = fluid.layers.rnn(cell=cell1, inputs=r_data, is_reverse=True)
    v1 = fluid.layers.concat(input=[fw1,bw1], axis=2)

    cell2 = fluid.layers.LSTMCell(hidden_size=2*vector_shape)
    fw2, _ = fluid.layers.rnn(cell=cell2, inputs=v1)
    bw2, _ = fluid.layers.rnn(cell=cell2, inputs=v1, is_reverse=True)
    v2 = fluid.layers.concat(input=[fw2,bw2], axis=2)

    

    fc = fluid.layers.fc(input=v2, size=100, act='relu')  
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)

    prediction = fluid.layers.fc(input=drop, size=2, act="softmax")  
       
    return prediction

def NP_CBiR(data):
    conv_1 = fluid.layers.conv2d(input=data, num_filters=50, filter_size=(5,vector_shape), act="tanh")
    conv_1 =fluid.layers.batch_norm(input=conv_1)
    length = conv_1.shape[2]
    reshape1 = fluid.layers.reshape(x=conv_1, shape=[-1,50,length])
    cell1 = fluid.layers.GRUCell(hidden_size=50, activation=fluid.layers.relu)
    fw1, _ = fluid.layers.rnn(cell=cell1, inputs=fluid.layers.transpose(reshape1, perm=[0,2,1])) 
    bw1, _ = fluid.layers.rnn(cell=cell1,inputs=fluid.layers.transpose(reshape1, perm=[0,2,1]),is_reverse=True)
    v1 = fluid.layers.concat(input=[fw1,bw1], axis=2)

    cell2 = fluid.layers.LSTMCell(hidden_size=100)
    fw2, _ = fluid.layers.rnn(cell=cell2, inputs=v1)
    bw2, _ = fluid.layers.rnn(cell=cell2, inputs=v1, is_reverse=True)
    v2 = fluid.layers.concat(input=[fw2,bw2], axis=2) 
   

    fc1= fluid.layers.fc(input=v2, size=100, act='relu') 
    fc1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
     
    prediction = fluid.layers.fc(input=fc1, size=2, act="softmax") 
 
    return prediction

final_test_acc = []
final_test_Sn = []
final_test_Sp =[]
final_mcc = []
final_auc = []
tprs = []
mean_fpr=np.linspace(0,1,100)

Dataset = np.loadtxt("work/Ce-new-list.txt", dtype=np.str, delimiter=' ')
np.random.shuffle(Dataset)
n_splits=10
sKF = StratifiedKFold(n_splits=n_splits, shuffle=False)
i = 0

num_epochs = 10
print('Start %d-fold cross validation'%(n_splits))
for train_index, test_index in sKF.split(Dataset[:,0],Dataset[:,1]):
    i +=1
    print('The %d fold' %(i))
    train_program = fluid.Program()
    startup_prog = fluid.Program() 
    with fluid.program_guard(train_program, startup_prog):
        seq = fluid.data(name='seq', shape=[-1,144,100], dtype='float32')  
        label = fluid.data(name='label', shape=[-1,1], dtype='int64')        
        with fluid.unique_name.guard():
            predict = CNN(seq)   #predict = CNN(seq) or predict = NP_CBiR(seq)
            cost = fluid.layers.cross_entropy(input=predict, label=label)  
            avg_cost = fluid.layers.mean(cost)
            acc = fluid.layers.accuracy(input=predict, label=label)           
            test_program = train_program.clone(for_test=True)       
            optimizer = fluid.optimizer.AdamaxOptimizer(learning_rate=0.0001,
                                                         regularization=fluid.regularizer.L2Decay(regularization_coeff=0.01))                                     
            opts = optimizer.minimize(avg_cost)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)    
    feeder = fluid.DataFeeder(place=place, feed_list=[seq, label])
    model_save_dir = str("/home/aistudio/model/dna%s.inference.model" %i)
    train_dna = Dataset[train_index]
    test_dna = Dataset[test_index]
    Batch_size = 64           
    train_reader = paddle.batch(seqdata_reader(train_dna, seqDict), batch_size=Batch_size)
    train_label = list(train_dna[:,1])
    train_label = list(map(int, train_label))
    test_reader = paddle.batch(seqdata_reader(test_dna, seqDict), batch_size=Batch_size)
    test_label = list(test_dna[:,1])
    test_label = list(map(int, test_label))
    predict_label = []
    score_test = []
    stop_train = False
    
    for pass_id in range(num_epochs):
        Train_predict=[]
        for step_id, data in enumerate(train_reader()):
            train_out = exe.run(program=train_program,          
                                        feed=feeder.feed(data),                 
                                        fetch_list=[predict])         
                             
            for item in train_out:
                for it in item:
                    Train_predict.append(np.argmax(it))                         
        tn, fp, fn, tp = confusion_matrix(train_label, Train_predict).ravel()
        train_acc = (tn + tp)/(tn + fp + fn + tp ) 
        train_Sn = tp/(fn+tp)
        train_Sp = tn/(fp+tn)
        train_mcc = (tp*tn-fp*fn)/pow(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),0.5)   
        print('Pass:%d, train_Accuracy:%0.4f, train_Sn:%0.4f, train_Sp:%0.4f, train_mcc:%0.4f' 
                                                   % (pass_id, train_acc, train_Sn, train_Sp, train_mcc))
        
    for step, data in enumerate(test_reader()):                         
        out = exe.run(program=test_program,          
                                        feed=feeder.feed(data),                  
                                        fetch_list=[predict])          
        for item in out:
            for it in item:
                score_test.append(it[1])
                predict_label.append(np.argmax(it))
    fpr, tpr, thresholds = roc_curve(test_label,score_test)
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc = auc(fpr,tpr)            
    tn, fp, fn, tp = confusion_matrix(test_label, predict_label).ravel()
    test_acc = (tn + tp)/(tn + fp + fn + tp ) 
    test_Sn = tp/(fn+tp)
    test_Sp = tn/(fp+tn)
    mcc = (tp*tn-fp*fn)/pow(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),0.5) 
    final_test_acc.append(test_acc)
    final_test_Sn.append(test_Sn)
    final_test_Sp.append(test_Sp)
    final_mcc.append(mcc)
    final_auc.append(roc_auc)
    print('test_Accuracy:%0.4f, test_Sn:%0.4f, test_Sp:%0.4f, mcc:%0.4f, roc_auc:%0.4f' 
                                                   % (test_acc, test_Sn, test_Sp, mcc, roc_auc))
    print("confusion matrix:\n"+str (confusion_matrix(test_label, predict_label)))
    print('---------------------------------------------------')
               
Final_test_acc = (sum(final_test_acc) / len(final_test_acc)) 
Final_test_Sn = (sum(final_test_Sn) / len(final_test_Sn)) 
Final_test_Sp = (sum(final_test_Sp) / len(final_test_Sp)) 
Final_mcc =  (sum(final_mcc) / len(final_mcc))
Final_auc = (sum(final_auc)/len(final_auc))                            
print('Final_test_Accuracy:%0.4f' % (Final_test_acc))   
print('Final_test_Sn:%0.4f' % (Final_test_Sn)) 
print('Final_test_Sp:%0.5f' % (Final_test_Sp)) 
print('Final_mcc:%0.4f' % (Final_mcc)) 
print('Final_AUC:%0.4f' % (Final_auc))

print('%d-fold cross validation is finshed.'%(n_splits))