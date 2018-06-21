#-*-coding:utf-8-*-
'''
药品使用记录-tfidf
'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

input=open("dataset/medicine.csv",encoding="utf-8")
corpus=[]
for line in input.readlines():
    corpus.append(line.strip("\n"))
vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
tfidf = transformer.fit_transform(
    vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    print("-------这里输出第", i, "个样本的词语tf-idf权重------")
    for j in range(len(word)):
        print(word[j], weight[i][j])