from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import lit, concat
from  pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words, wordnet 
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from pyspark import SparkContext, SparkConf
from elephas.spark_model import SparkMLlibModel
from pyspark.mllib.regression import LabeledPoint
import eel

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('omw')



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="dataset name", default="amazon_reviews_baby_100k.tsv", type=str)
parser.add_argument("--model", "-m", help="1:review_text or 2:review_text+DB", default=2, type=int)
parser.add_argument("--gui", "-g", help="Use the GUI or not", action="store_true")
useGui = 0

### Connect to spark and load dataset


### Dropping rows with na values and creating target variable column
def condition(r):
    if (r == 1):
        #label="negative"
        label=0
    else:
        #label="positive"
        label=1
    return label


#### Text processing on the reviews with NLTK:
#1) tokenize the text.
#2) Filter to keep only english words
#3) Remove stop words
#4) Lemmatization

words = set(nltk.corpus.words.words())
stopwords = stopwords.words('english')
stemmer = PorterStemmer()
lemmatize = WordNetLemmatizer()
wordnetVerb= wordnet.VERB

def text_preprocessing(text):
    text = ''.join([i for i in text if not i.isdigit()])
    tokens=nltk.tokenize.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords]
    lemmatized_words = []
    for w in tokens:
        rootWord = lemmatize.lemmatize(w, wordnetVerb)
        lemmatized_words.append(rootWord)
    stemming_words = []
    for w in lemmatized_words:
        rootWord = stemmer.stem(w)
        stemming_words.append(rootWord)
    english_tokens = []
    for token in stemming_words:
        if token.lower() in words:
            english_tokens.append(token)
    return english_tokens


def main(args):
    dataset = args.dataset
    conf = SparkConf().setAppName('Elephas_App')
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()
    df_original = spark.read.option("header", "true") \
        .option("delimiter", "\t") \
        .option("inferSchema", "true").csv("hdfs://10.0.1.111:9000/"+dataset)

    sentiment_udf = udf(lambda x: condition(x), IntegerType())
    df = df_original.na.drop()
    df = df.withColumn('Sentiment',sentiment_udf(df['star_rating']))
    df = df.drop('star_rating')

    if args.model == 1:
        if useGui == 1:
            print("### Building model 1")
            stat = "Building Model 1...."
            statUpdate(stat)
        df_model1 = df.select(df['review_body'], df['Sentiment'].alias('label'))
        preprocessing_udf = udf(lambda x: text_preprocessing(x), ArrayType(StringType()))
        df_model1 = df_model1.withColumn('keywords',preprocessing_udf(df_model1['review_body']))
        df_model1 = df_model1.na.drop()
        #df_model1.show(10)
    
    if args.model == 2:
        print("### Building model 2")
        if useGui == 1:
            stat = "Building Model 2..."
            statUpdate(stat)
        df_model2 = df.select(df['helpful_votes'], df['total_votes'], concat(df['review_headline'], lit('. '), df['review_body']).alias('reviews'), df['Sentiment'].alias('label'))
        df_model2 = df_model2.na.drop()
        preprocessing_udf = udf(lambda x: text_preprocessing(x), ArrayType(StringType()))
        df_model2 = df_model2.withColumn('keywords',preprocessing_udf(df_model2['reviews']))
        df_model2 = df_model2.na.drop()

    #Create training data and labels
    if args.model == 1:
        X_train, X_test = df_model1.randomSplit([0.8, 0.2], seed=1)

    if args.model == 2:
        X_train, X_test = df_model2.randomSplit([0.8, 0.2], seed=1)

    print("Term Frequency of Keywords")
    if useGui == 1:
        stat = "Calculating Term Frequency of Keywords..."
        statUpdate(stat)
    cv = CountVectorizer()
    cv.setInputCol("keywords")
    cv.setOutputCol("vectors")
    model_cv = cv.fit(X_train)
    model_cv.setInputCol('keywords')
    X_train = model_cv.transform(X_train)
    X_test = model_cv.transform(X_test)

    if args.model == 1:
        assembler = VectorAssembler(inputCols=['vectors'], outputCol='features')
    if args.model == 2:
        assembler = VectorAssembler(inputCols=['helpful_votes', 'total_votes','vectors'], outputCol='features')

    X_train = assembler.transform(X_train)
    X_test = assembler.transform(X_test)
    X_train = X_train.select(X_train.features, X_train.label)    
    X_test = X_test.select(X_test.features, X_test.label)
    print("Term Frequency of Keywords ... DONE")
    if useGui == 1:
        stat = "Term Frequency of Keywords ... DONE"
        statUpdate(stat)
    


    # Feature selection
    print("Feature Selection: Select 100 best keywords")
    if useGui == 1:
        stat = "Feature Selection: Select 100 best keywords..."
        statUpdate(stat)
    selector = UnivariateFeatureSelector(outputCol="selectedFeatures")
    selector.setFeatureType("categorical").setLabelType("categorical").setSelectionThreshold(100)
    model_select = selector.fit(X_train)
    model_select.setFeaturesCol("features")
    X_train = model_select.transform(X_train)
    X_test = model_select.transform(X_test)
    X_train_kbest = X_train.drop('features')
    X_test_kbest = X_test.drop('features')
    print("Feature Selection ... Done")
    if useGui == 1:
        stat = "Feature Selection ... Done"
        statUpdate(stat)
    

    ##### Classification
    ## Neural Network
    model = Sequential()
    model.add(Dense(14, input_dim=100, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    learning_rate = 0.001
    #model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    
    X = X_train_kbest.withColumnRenamed('selectedFeatures','features')
    rdd = X.rdd
    lp_rdd = rdd.map(lambda row: LabeledPoint(row['label'], row['features'].toArray()))
    spark_model = SparkMLlibModel(model, frequency='batch', mode='hogwild')
    print("# Neural Network (distributed training)")
    if useGui == 1:
        stat = " Neural Network (distributed training) Started..."
        statUpdate(stat)
    spark_model.fit(lp_rdd, epochs=10, batch_size=32, verbose=0, validation_split=0.2, categorical=True, nb_classes=2)

    print("# Neural Network (distributed training) ... DONE")
    if useGui == 1:
        stat = " Neural Network (distributed training) DONE..."
        statUpdate(stat)


    ## Random Forest
    print("##### Build Classifiers ######")
    print("# Random Forest")
    if useGui == 1:
        stat = " Building Classifier: Random Forest..."
        statUpdate(stat)
    rf = RandomForestClassifier(featuresCol='selectedFeatures', numTrees=3, maxDepth=2, labelCol="label", seed=42, leafCol="leafId")
    rf.getMinWeightFractionPerNode()
    model_rf = rf.fit(X_train_kbest)
    model_rf.setFeaturesCol("selectedFeatures")
    #model_rf.setRawPredictionCol("newRawPrediction")
    y_predict_RF = model_rf.transform(X_test_kbest)
    print("# Random Forest ... DONE")
    if useGui == 1:
        stat = "Random Forest... DONE"
        statUpdate(stat)
    ## Naive Bayes
    #nb = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol='selectedFeatures')
    print("# Naive Bayes")
    if useGui == 1:
        stat = " Building Classifier: Naive Bayes..."
        statUpdate(stat)
    nb = NaiveBayes(smoothing=1.0, modelType="gaussian", featuresCol='selectedFeatures')
    model_NB = nb.fit(X_train_kbest)
    model_NB.setFeaturesCol("selectedFeatures")
    y_predict_NB = model_NB.transform(X_test_kbest)
    print("# Naive Bayes ... DONE")
    if useGui == 1:
        stat = "Naive Bayes... DONE"
        statUpdate(stat)
    

    ## Logistic Regression
    print("# Logistic Regression")
    if useGui == 1:
        stat = " Building Classifier: Logistic Regression..."
        statUpdate(stat)
    lr = LogisticRegression(featuresCol='selectedFeatures')
    lr.setMaxIter(10)
    lr.setRegParam(0.01)
    model_LR = lr.fit(X_train_kbest)
    model_LR.setFeaturesCol("selectedFeatures")
    y_predict_LR = model_LR.transform(X_test_kbest)
    print("# Logistic Regression ... DONE")
    if useGui == 1:
        stat = "Logistic Regression ... DONE..."
        statUpdate(stat)

    
    #### Metrics
    print("Perfomance Comparison (RF, NB, LR, NN): F1_score")
    if useGui == 1:
        stat = "Performance Comparison..."
        statUpdate(stat)
    evaluator = MulticlassClassificationEvaluator()
    evaluator.setPredictionCol("prediction")
    #evaluator.evaluate(y_predict_RF, {evaluator.metricName: "accuracy"})
    f1score_RF = evaluator.evaluate(y_predict_RF, {evaluator.metricName: "f1"})
    print(f"Random Forest: {f1score_RF:.2f}")
    f1score_NB = evaluator.evaluate(y_predict_LR, {evaluator.metricName: "f1"})
    print(f"Naive Bayes: {f1score_NB:.2f}")
    f1score_LR = evaluator.evaluate(y_predict_NB, {evaluator.metricName: "f1"})
    print(f"Logistic Regression: {f1score_LR:.2f}")

    X_test_kbest = X_test_kbest.withColumnRenamed('selectedFeatures','features')
    pandaDf = X_test_kbest.toPandas()
    y_test =  pandaDf['label'].values
    X_test_df = pandaDf['features'].apply(lambda x: np.array(x.toArray())).to_numpy().reshape(-1,1)
    X_test_df = np.apply_along_axis(lambda x: x[0], 1, X_test_df)
    from keras.utils.np_utils import to_categorical
    y_test = to_categorical(y_test, num_classes = 2)
    scores = model.evaluate(x=X_test_df, y=y_test, verbose=0)
    precision = scores[2]
    recall = scores[3]
    f1score_NN = 2 * (precision * recall) / (precision + recall)
    print(f'Neural Network: {f1score_NN:.2f}')
    if useGui == 1:
        stat = "ALL DONE..."
        statUpdate(stat)
        return f1score_NN, f1score_RF, f1score_LR, f1score_NB


if __name__ == '__main__':
    args = parser.parse_args()
    if args.gui:
        useGui = 1

        eel.init("web")

        #Get inputs values from the GUI to python
        @eel.expose    
        def getInputValues(dataset, model):
            args.dataset = dataset
            args.model = model
            f1score_NN, f1score_RF, f1score_LR, f1score_NB = main(args)
            sendResultToWeb(f1score_NN, f1score_RF, f1score_LR, f1score_NB)

        #Send results from python to GUI 
        def sendResultToWeb(f1score_NN, f1score_RF, f1score_LR, f1score_NB):      #(dl=deep learning, rf=random forest, nb=naive bayes)
            eel.pythonToGui(f1score_NN, f1score_RF, f1score_LR, f1score_NB)       #Call a Javascript function and pass the results to it

        #Send Status Update to GUI
        def statUpdate(status):
            eel.statusUpdate(status)

        eel.start("index.html")

    main(args)

