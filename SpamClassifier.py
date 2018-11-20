from configuration import *
from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

from pyspark.mllib.classification import SVMWithSGD, SVMModel
import numpy as np
import pandas as pd

sc = SparkContext("local[2]", "WeiboSpammerClassification")


def divide(a, b):
    if b == 0:
        return a
    return a / b

class SpamClassifier():

    def loadData(self):
        self.weibo_data = []
        with open(ARRAY_PATH, 'r') as r:
            for line in r:
                single = list(map(float, line.strip('\n').split(" ")))
                self.weibo_data.append(single)
        self.weibo_data = pd.DataFrame(self.weibo_data)

        with open(HEADER_PATH, 'r') as r:
            title = r.readline().strip('\n').split(" ")
            self.weibo_data.columns = title
        with open(Y_PATH, 'r') as r:
            self.weibo_data.insert(0, 'label', r.read().splitlines())

    def preprocessing(self):
        # 对平均每天微博字数与平均每条微博@别人次数的缺失值，使用其同类用户的平均值进行处理
        cols = [col for col in self.weibo_data.columns if col in ['平均每天微博字数', '平均每条微博@别人次数']]
        gp_col = 'label'        # 分组的列
        df_mean = self.weibo_data.groupby(gp_col)[cols].mean()
        # print(df_mean)
        df_na = self.weibo_data[cols].isnull()        # 依次处理每一列
        for col in cols:
            na_series = df_na[col]
            names = list(self.weibo_data.loc[na_series, gp_col])
            t = df_mean.loc[names, col]
            t.index = self.weibo_data.loc[na_series, col].index
            self.weibo_data.loc[na_series, col] = t# 相同的index进行赋值

    # 近15天用户每天平均发博数量
    def calculate_average_count(self):
        average_count = []
        with open(ACTIVE_STAT_PATH , 'r') as r:
            for line in r:
                nums = line.strip('\n').split(" ")
                nums = list(map(int, nums))
                average_count.append(np.mean(nums))
        self.weibo_data['平均每天发博数量'] = average_count

    # 粉丝关注比，即粉丝数/关注数
    def calculate_FF(self):
        self.weibo_data['粉丝关注比'] = self.weibo_data.apply(lambda x: divide(x['粉丝数'], x['关注数']), axis=1)

    # 原创微博占比：即原创微博数量 / 微博数量
    def calculate_origin(self):
        self.weibo_data['原创微博占比'] = self.weibo_data.apply(lambda x: divide(x['原创微博数量'], x['微博数量']), axis=1)

    def bayes(self):
        model = NaiveBayes.train(self.training, 1.0)

        # Make prediction and test accuracy.
        predictionAndLabel = self.test.map(lambda p: (model.predict(p.features), p.label))
        accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / self.test.count()
        print('model accuracy {}'.format(accuracy))

    def decision_tree(self):
        model = DecisionTree.trainClassifier(self.training, numClasses=2, categoricalFeaturesInfo={},
                                             impurity='gini', maxDepth=5, maxBins=32)

        # Evaluate model on test instances and compute test error
        predictions = model.predict(self.test.map(lambda x: x.features))
        labelsAndPredictions = self.test.map(lambda lp: lp.label).zip(predictions)
        accuracy = 1.0 * labelsAndPredictions.filter(
            lambda lp: lp[0] == lp[1]).count() / float(self.test.count())
        print('model accuracy {} '.format(accuracy))
        print('Learned classification tree model:')
        print(model.toDebugString())

    def svm(self):
        model = SVMWithSGD.train(self.training, iterations=100)

        # Evaluating the model on training data
        labelsAndPreds = self.test.map(lambda p: (p.label, model.predict(p.features)))
        accuracy = 1.0 * labelsAndPreds.filter(lambda lp: lp[0] == lp[1]).count() / float(self.test.count())
        print("model accuracy {} ".format(accuracy))

    def oneShotTraining(self):
        self.loadData()
        self.preprocessing()
        self.calculate_average_count()
        self.calculate_FF()
        self.calculate_origin()

        cols = [col for col in self.weibo_data.columns if col in FEATURE_LIST]
        features = self.weibo_data[cols]
        data = []
        for index, row in features.iterrows():
            data.append(LabeledPoint(int(row[0]), row[1:9].tolist()))
        self.training, self.test = sc.parallelize(data).randomSplit([0.7, 0.3])
        # self.bayes()
        self.decision_tree()
        # self.svm()

classifier = SpamClassifier()
classifier.oneShotTraining()