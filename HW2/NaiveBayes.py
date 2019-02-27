# -*- coding: utf-8 -*-
"""Copy of Document Classification using Naive Bayes Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a8tT82xITBJ2jCqnjkJZds_ZxVnaheXI

# **Session 7: Probability and Statistics for AI & Machine Learning II**
## Document Classification using Naive Bayes Classifier.

## PY599 (Fall 2018): Applied Artificial Intelligence
## NC State University
###Dr. Behnam Kia
### https://appliedai.wordpress.ncsu.edu/


**Disclaimer**: Please note that these codes are simplified version of the algorithms, and they may not give the best, or expected performance that you could possibly get from these algorithms. The aim of this notebook is to help you understand the basics and the essence of these algorithms, and experiment with them. These basic codes are not deployment-ready or free-of-errors for real-world applications. To learn more about these algorithms please refer to text books that specifically study these algorithms, or contact me. - Behnam Kia

# Dataset

Method 1: You can download dataset from Keras. In this dataset the words are replaced by a unique number. According to Keras' website, "Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word."
"""

from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

x_train.shape

print(x_train[0])

import numpy as np

#p(c)
prob_c=dict()
prob_c[1]=np.sum(y_train)/len(y_train)#12500
prob_c[0]=1-prob_c[1]#12500

#calculate the frequency of each word in negative reviews and positive reviews
count_word_frequency=dict()
for i in range(0,x_train.shape[0]):
    for word in x_train[i,]:
        if word not in count_word_frequency:
           count_word_frequency[word]=[0,0]
        if y_train[i]==0:
           count_word_frequency[word][0] += 1
        if y_train[i]==1:
           count_word_frequency[word][1] += 1
  
# calculate the positive word count and negative word count
sum=[0,0]
for word in count_word_frequency:
    sum[0] += count_word_frequency[word][0]
    sum[1] += count_word_frequency[word][1]
import math

# test_review
N=len(count_word_frequency)
result=np.zeros(x_test.shape[0])
#print(x_test.shape[0])

for j in range(0,x_test.shape[0]):
   log_test_posterior_0 = math.log(prob_c[0])
   log_test_posterior_1 = math.log(prob_c[1])
   current_review=dict()
   for word in x_test[j,]:
     #print(word)
     if word not in current_review:
        current_review[word] = 0
     else:
        current_review[word] += 1
   for word in current_review.keys():
     if word not in count_word_frequency:# not in postive and not in negative
        log_test_posterior_0 += math.log(1/(sum[0]+N))
        log_test_posterior_1 += math.log(1/(sum[1]+N)) 
     else:
        log_test_posterior_0 += current_review[word]*math.log((float(count_word_frequency[word][0]+1))/(sum[0]+N))
        log_test_posterior_1 += current_review[word]*math.log((float(count_word_frequency[word][1]+1))/(sum[1]+N))
   if log_test_posterior_0<log_test_posterior_1:
      result[j]=1
   else:
      result[j]=0
      
#print(np.sum(np.absolute(result-y_test)))
error_rate = np.sum(np.absolute(result-y_test))/len(y_test)
print('error_rate is %f'%(error_rate))

"""Method 2: You can download the original dataset with readible reviews. Please go to: 
http://ai.stanford.edu/~amaas/data/sentiment/

and download "Large Movie Review Dataset v1.0."

There are many different methods to upload dataset to colab. 
One method is to download the dataset to your local computer, then upload it to colab and then unzip it. Please see the code below:
"""

from google.colab import files

uploaded = files.upload()
!tar xzvf aclImdb_v1.tar.gz >/dev/null

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=10,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

print(x_train)

#step 1: combine all the postive review in one string
#step 2: split str to a list of word and count words
import os
import re
import collections


positive_reviews=os.listdir("aclImdb/train/pos")
negative_reviews=os.listdir("aclImdb/train/neg")

#positive_train
str=""
for review in positive_reviews:
    f=open('aclImdb/train/pos/'+review).read()
    str = str + f
wordListPos = re.sub("[^\w]"," ",str).split() #list
cntPos=collections.Counter()
for word in wordListPos:
  cntPos[word] +=1
 
#negative_train  
str=""
for review in negative_reviews:
    f=open('aclImdb/train/neg/'+review).read()
    str = str + f
wordListNeg = re.sub("[^\w]"," ",str).split() #list
cntNeg=collections.Counter()
for word in wordListNeg:
  cntNeg[word] +=1
  
#pos_count and negative_count  
Pos_count=0
for key in cntPos:
   Pos_count += cntPos[key]
Neg_count=0
for key in cntPos:
   Neg_count += cntNeg[key]
  
#Vocabulary
Voc_f=open('aclImdb/imdb.vocab')
Voc=list()
for word in Voc_f:
  Voc.append(word) #number of words:89527

  
# test
test_reviews_neg=os.listdir("aclImdb/test/neg")
test_reviews_pos=os.listdir("aclImdb/test/pos")

#analysis negative test
result = np.zeros(len(test_reviews_neg))   
for i in range(0,len(test_reviews_neg)):
    test_review = test_reviews_neg[i]
    #print(test_review)
    f=open("aclImdb/test/neg/" + test_review).read()
    wordthisReview = re.sub("[^\w]"," ",f).split() 
    cntReview = collections.Counter()
    ReviewKeys = cntReview.keys()
    for word in wordthisReview:
       cntReview[word] +=1
    LoglikelihoodPos = math.log(1/2)# train_pos=train_neg
    for word in ReviewKeys:
       LoglikelihoodPos += cntReview[word]*math.log((float(cntPos[word])+1)/(Pos_count+len(Voc)))
    LoglikelihoodNeg = math.log(1/2)
    for word in ReviewKeys:
       LoglikelihoodNeg += cntReview[word]*math.log((float(cntNeg[word])+1)/(Neg_count+len(Voc)))
    if LoglikelihoodPos > LoglikelihoodNeg:
       result[i] = 1#pos for 1

error_neg = np.sum(np.absolute(result - np.zeros(len(test_reviews_neg))))

result = np.ones(len(test_reviews_pos))   
for i in range(0,len(test_reviews_pos)):
    test_review = test_reviews_pos[i]
    #print(test_review)
    f=open("aclImdb/test/pos/" + test_review).read()
    wordthisReview = re.sub("[^\w]"," ",f).split() 
    cntReview = collections.Counter()
    ReviewKeys = cntReview.keys()
    for word in wordthisReview:
       cntReview[word] +=1
    LoglikelihoodPos = math.log(1/2)# train_pos=train_neg
    for word in ReviewKeys:
       LoglikelihoodPos += cntReview[word]*math.log((float(cntPos[word])+1)/(Pos_count+len(Voc)))
    LoglikelihoodNeg = math.log(1/2)
    for word in ReviewKeys:
       LoglikelihoodNeg += cntReview[word]*math.log((float(cntNeg[word])+1)/(Neg_count+len(Voc)))
    if LoglikelihoodPos < LoglikelihoodNeg:
       result[i] = 0#neg for 0
error_pos = np.sum(np.absolute(result - np.ones(len(test_reviews_pos))))
error_rate = (error_neg+error_pos)/(len(test_reviews_pos)+len(test_reviews_neg))
print('error_rate is %f'%(error_rate))