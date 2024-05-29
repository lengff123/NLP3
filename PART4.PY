import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import random


def load_stopwords(file_path):
    stop_words = []
    with open('cn_stopwords.txt', "r", encoding="utf-8", errors="ignore") as f:
        stop_words.extend([word.strip('\n') for word in f.readlines()])
    return stop_words

def preprocess_corpus( text,cn_stopwords):
    for tmp_char in cn_stopwords:
        text = text.replace(tmp_char, "")             
    return text 



#import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
if __name__ == '__main__':
    stopwords_file_path = 'cn_stopwords.txt'
    cn_stopwords = load_stopwords(stopwords_file_path)          
    corpus_dict = {}  # 假设这是您的语料库字典
    book_titles_list = "白马啸西风,碧血剑,飞狐外传,连城诀,鹿鼎记,三十三剑客图,射雕英雄传,神雕侠侣,书剑恩仇录,天龙八部,侠客行,笑傲江湖,雪山飞狐,倚天屠龙记,鸳鸯刀,越女剑"#
    for book_title in book_titles_list.split(','):
        book_title = book_title.strip()  # 去除可能存在的多余空白字符
        file_path='jyxstxtqj_downcc.com\{}.txt'.format(book_title)
        merged_content = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            merged_content += f.read()
        # 保存合并后的内容到新的文本文件
        merged_content=preprocess_corpus( merged_content,cn_stopwords)
        output_file_path = 'fr\{}.txt'.format(book_title)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
            

        # 分词并准备训练数据
        sentences = merged_content.split()

        # 训练Word2Vec模型
        model = Word2Vec(sentences, window=5, min_count=1, sg=1)

        # 保存模型
        model.save("word2vec_model.bin")

        # 加载模型以便后续使用（可选）
        # model = Word2Vec.load("word2vec_model.bin")

        # 验证词向量的有效性
        # 计算词向量间的相似度
        print(model.wv.similarity('剑', '武功'))  # 假设'郭靖'和'黄蓉'是小说中的两个人物名
        print(model.wv.most_similar('侠'))

        words = list(model.wv.index_to_key)[:1000]  # 取前1000个词
        word_vectors = model.wv[words]

        kmeans = KMeans(n_clusters=5)
        kmeans.fit(word_vectors)

        # 可视化（简化展示）
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(word_vectors)
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=kmeans.labels_)
        #for i, word in enumerate(words):
        #    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
        plt.show()

    
        # 假设你有段落数据
        paragraph_vectors = [model.wv[jieba.lcut(paragraph)].mean(axis=0) for paragraph in sentences]
        # 计算两段落间余弦相似度作为语义关联度
        similarity_between_paragraphs = np.dot(paragraph_vectors[0], paragraph_vectors[1]) / (np.linalg.norm(paragraph_vectors[0]) * np.linalg.norm(paragraph_vectors[1]))

                