import flask
from flask import Flask, jsonify, request,url_for
import pickle
import numpy as np
import pandas as pd
from sklearn import linear_model
import re
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



app = Flask(__name__, template_folder="templates")
vect = pickle.load(open('tfid.nav', 'rb'))
classifier = pickle.load(open('gnb.nav','rb'))
lemmatizer = WordNetLemmatizer()


def preprocess(raw_mess):
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',' ', raw_mess)
    text = re.sub('@[^\s]+',' ', text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(i) for i in text if not i in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    review_text = preprocess(to_predict_list['review_text'])
    encoded = vect.transform([review_text])
    pred = classifier.predict(encoded)

    if pred==1:
        prediction = "Negative"
    else:
        prediction = "Positive"

    return flask.render_template('predict.html',prediction = prediction)

if __name__=='__main__':
    app.run(debug=True)
