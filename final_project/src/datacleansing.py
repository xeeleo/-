import pandas as pd
import jieba as jb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


# 加载停用词：
with open("../Data/hit_stopwords.txt", "r", encoding="utf-8") as fr:
    stopwords = fr.read().splitlines()
print(stopwords)


data_train = pd.read_csv('../Data/train.csv')
data_train.fillna('', inplace=True)  # 在原对象的基础上修改，将空闲处填上’‘
print(data_train)
data_test = pd.read_csv('../Data/test.csv')
data_test.fillna('', inplace=True)
print(data_test)

train_data = []  # 存放训练数据
train_label = []  # 存放训练数据的结果
for i in range(0, len(data_train)):
    train_data.append(jb.lcut(data_train['content'][i]) + jb.lcut(data_train['comment_all'][i]))
    train_label.append(data_train['label'][i])

# 清洗训练数据：
for i in train_data:
    for every_stop_word in stopwords:
        if i.__contains__(every_stop_word):
            i.remove(every_stop_word)


#  将其变成一维：
train_data1D = []
for i in train_data:
    cn = " ".join(j for j in i)
    train_data1D.append(cn)

train_data_fit = np.array(train_data1D)
train_label_fit = np.array(train_label)


test_data = []
for i in range(0, len(data_test)):
    test_data.append(jb.lcut(data_test['content'][i]) + jb.lcut(data_test['comment_all'][i]))


# 清洗测试数据：
for i in test_data:
    for every_stop_word in stopwords:
        if i.__contains__(every_stop_word):
            i.remove(every_stop_word)


# 将二维转换为一维
test_data1D = []
for i in test_data:
    cn = " ".join(j for j in i)
    test_data1D.append(cn)
test_data_fit = np.array(test_data1D)


# 模型使用1：CountVectorizer + MultinomialNB
vec = CountVectorizer()
train_sample1 = vec.fit_transform(train_data_fit)
test_sample1 = vec.transform(test_data_fit)
clf1 = MultinomialNB(alpha=0.001)
clf1.fit(train_sample1, train_label_fit)
predict1 = clf1.predict(test_sample1)  #
print(predict1)
np.savetxt("../Data/result1.txt", predict1, fmt="%d")


# 模型使用2： TfidfTransformer + MultinomialNB
tfidf = TfidfTransformer()
train_sample = tfidf.fit_transform(train_sample1)
test_sample = tfidf.transform(test_sample1)  #
clf = MultinomialNB(alpha=0.001)
clf.fit(train_sample, train_label_fit)
predict = clf.predict(test_sample)  #
print(predict)
np.savetxt("../Data/result.txt", predict, fmt="%d")
