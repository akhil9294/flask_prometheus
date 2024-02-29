from flask import Flask, jsonify, Response
#from fun import armstrong

import prometheus_client 
from prometheus_client.core import CollectorRegistry
from prometheus_client import Summary, Gauge, Histogram, Counter

import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from flask import Flask, jsonify, request

import pickle
from joblib import dump, load

app = Flask(__name__)

_INF = float("inf")
graphs1 = {}
graphs2 = {}
graphs1 ['c'] = Counter('python_request_operations_total', 'The total number of processed requests')
graphs1 ['h'] = Histogram('python_request_duration_seconds', 'Histogram for the duration in seconds.', buckets=(1, 2, 5, 6, 10, _INF))
graphs2 ['c'] = Counter('python_request_operations_total1', 'The total number of processed requests with sleep')
graphs2 ['h'] = Histogram('python_request_duration_seconds1', 'Histogram for the duration in seconds with sleep', buckets=(1, 2, 5, 6, 10, _INF))
#graphs ['h'] = Histogram('python_request_duration_seconds', 'Histogram for the duration in seconds.', buckets=Buckets.exponential(10, 10, 5))#
#graphs ['h'] = Histogram('python_request_duration_seconds', 'Histogram for the duration in seconds.')


@app.route('/')
def hello_world():
    return {'Welcome message':'Hello, World!'}


@app.route('/diabeties_classifier')
def diabeties_classifier():
    start = time.time()
    Pregnancies = int(request.args.get('Pregnancies'))
    Glucose = int(request.args.get('Glucose'))
    BloodPressure = int(request.args.get('BloodPressure'))
    SkinThickness = int(request.args.get('SkinThickness'))
    Insulin = int(request.args.get('Insulin'))
    BMI = int(request.args.get('BMI'))
    DiabetesPedigreeFunction = float(request.args.get('DiabetesPedigreeFunction'))
    Age = int(request.args.get('Age'))
    X_test = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    bin_scaler =load('std_scaler.bin')
    X_test = bin_scaler.transform(X_test)
    pickled_model = pickle.load(open('model_classifier.pkl', 'rb'))
    y_pred = pickled_model.predict(X_test)
    #time.sleep(Sleep_sec)
    end  = time.time()
    graphs1['h'].observe(end - start)
    if y_pred[0]==0:
        return 'Not Diabetic'
    else:
        return 'Diabetic'
 
@app.route('/diabeties_classifier_with_sleep')
def diabeties_classifier_with_sleep():
    Pregnancies = int(request.args.get('Pregnancies'))
    Glucose = int(request.args.get('Glucose'))
    BloodPressure = int(request.args.get('BloodPressure'))
    SkinThickness = int(request.args.get('SkinThickness'))
    Insulin = int(request.args.get('Insulin'))
    BMI = int(request.args.get('BMI'))
    DiabetesPedigreeFunction = float(request.args.get('DiabetesPedigreeFunction'))
    Age = int(request.args.get('Age'))
    Sleep_sec = int(request.args.get('Sleep_sec'))
    X_test = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    bin_scaler =load('std_scaler.bin')
    X_test = bin_scaler.transform(X_test)
    pickled_model = pickle.load(open('model_classifier.pkl', 'rb'))
    y_pred = pickled_model.predict(X_test)
    start = time.time()
    time.sleep(Sleep_sec)
    end  = time.time()
    graphs2['h'].observe(end - start)
    if y_pred[0]==0:
        return 'Not Diabetic'
    else:
        return 'Diabetic'

@app.route("/metrics")
def requests_counts():
    res = []
    for k,v in graphs1.items():
        res.append(prometheus_client.generate_latest(v))
    for k,v in graphs2.items():
        res.append(prometheus_client.generate_latest(v))
    return Response(prometheus_client.generate_latest(), mimetype="text/plain")





def armstrong (n):
    sum = 0
    order = len(str(n))
    copy_n = n
    while(n>0):
        digit = n%10
        sum += digit **order
        n = n//10
    if (sum == copy_n):
        print ("is an armstrong number")
        return True
    else:
        print (" is not an armstrong number")
        return False


@app.route('/a/<int:n>')
def armstrongg(n):
    start = time.time()
    a = str(armstrong(n))
    end  = time.time()
    graphs['h'].observe(end - start)
    return a

@app.route("/b/<int:n>")
def hello(n):
    start = time.time()
    # graphs['c'].inc()
    time.sleep(n*0.001)
    end  = time.time()
    graphs['h'].observe(end - start)
    return "sucess!"

@app.route("/metricss")
def requests_count():
    res = []
    print(graphs.items())
    for k,v in graphs.items():
        res.append(prometheus_client.generate_latest(v))
    return Response(res, mimetype="text/plain")



if __name__ == "__main__":
    app.run(debug=True)