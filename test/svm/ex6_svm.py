#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
使用自己的 svm 进行垃圾邮件分类。
"""


import re

import numpy as np
import scipy.io as scio
from nltk.stem import porter

from svm import SVC


def get_vocabs():
    vocabs = {}
    # 单词的总数
    n = 1899

    f = open('vocab.txt', 'r')
    for i in range(n):
        line = f.readline()
        idx = int(re.search(r'\d+', line).group(0)) - 1
        word = re.search(r'[a-zA-Z]+', line).group(0)
        vocabs[word] = idx
    f.close()

    return vocabs


def process_email(email_content: str):
    vocabs = get_vocabs()
    word_indices = []

    email_content = email_content.lower()
    email_content = re.sub(r'<[^<>]+>', ' ', email_content)
    email_content = re.sub(r'[0-9]+', 'number', email_content)
    email_content = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_content)
    email_content = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_content)
    email_content = re.sub(r'[$]+', 'dollar', email_content)

    print('\n==== Processed Email ====\n')

    l = 0
    tokens = re.split(r'[@$/#.-:&*+=\[\]?!\(\){},\'">_<;% ]', email_content)
    token_pattern = re.compile(r'[^a-zA-Z0-9]')
    stemmer = porter.PorterStemmer()
    for token in tokens:
        token = token_pattern.sub('', token)
        token = stemmer.stem(token)

        if len(token) < 1:
            continue

        if token in vocabs:
            word_indices.append(vocabs[token])

            if l + len(token) + 1 > 78:
                print()
                l = 0
            print(token + ' ', end='')
            l = l + len(token) + 1

    print('\n\n=========================')

    return word_indices


def email_features(word_indices: list):
    n = 1899
    m = len(word_indices)
    x = np.zeros((n,))
    for i in range(m):
        x[word_indices[i]] = 1

    return x


if __name__ == '__main__':
    'Part 1: Email Preprocessing'

    print('\nPreprocessing sample email (emailSample1.txt)')
    with open('emailSample1.txt', 'r') as f:
        file_contents = f.read()
        word_indices = process_email(file_contents)
        print(word_indices, '\n')

    input('Program paused. Press enter to continue.')

    'Part 2: Feature Extraction'

    print('\nExtracting features from sample email (emailSample1.txt)')
    file_contents = open('emailSample1.txt', 'r').read()
    word_indices = process_email(file_contents)
    features = email_features(word_indices)
    print('Length of feature vector: %d' % len(features))
    print('Number of non-zero entries: %d' % sum(features > 0))

    input('Program paused. Press enter to continue.')

    'Part 3: Train Linear SVM for Spam Classification'

    data = scio.loadmat('spamTrain.mat')
    X = (data['X']).astype(dtype=np.int)
    y = (data['y'].ravel()).astype(dtype=np.int)
    print('\nTraining Linear SVM (Spam Classification)')
    print('(this may take 1 to 2 minutes) ...')

    c = 0.1
    svc = SVC(labels=[1, 0], c=c, kernel='linear', tol=1e-5)
    svc.train(X, y)
    p = svc.predict(X)
    print(y[:20])
    print(p[:20])
    print('\nTraining Accuracy: %f' % (np.mean(p == y) * 100))

    'Part 4: Test Spam Classification'

    data = scio.loadmat('spamTest.mat')
    Xtest = (data['Xtest']).astype(dtype=np.int)
    ytest = (data['ytest'].ravel()).astype(dtype=np.int)

    print('\nEvaluating the trained Linear SVM on a test set ...')
    p = svc.predict(Xtest)
    print('Test Accuracy: %f' % (np.mean(p == ytest) * 100))

    input('Program paused. Press enter to continue.')

    'Part5: Test own email'

    filename = input('\n输入邮件文件名称（q 退出）：')
    while filename != 'q':
        print('测试邮件 %s 是否为垃圾邮件' % filename)
        with open(filename, 'r') as f:
            file_contents = f.read()
            word_indices = process_email(file_contents)
            x = email_features(word_indices)
            x = x.reshape((1, x.shape[0]))

            p = svc.predict(x)
            print('使用 SVM 处理邮件 %s，是否为垃圾邮件？' % filename, p == 1)

            filename = input('\n输入邮件文件名称（q 退出）：')
