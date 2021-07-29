from flask import Flask, render_template, request
import sklearn
import numpy as np
import pandas as pd
import pickle

flw = ['setosa', 'versicolor', 'virginica']

app = Flask(__name__)

@app.route("/")
def hello() :
    # return "Hi Hello how are you"
    return render_template("index.html")

@app.route("/redirect",methods = ['POST'])
def red() :
    if(request.method == "POST") :
        sepal_lenght = request.form["sepal_length"]
        sepal_width = request.form["sepal_width"]
        petal_length = request.form["petal_length"]
        petal_width = request.form["petal_width"]
        sepal_lenght = (float)(sepal_lenght)
        sepal_width = (float)(sepal_width)
        petal_length = (float)(petal_length)
        petal_width = (float)(petal_width)
        with open("model.pkl","rb") as rb1 : 
            md = pickle.load(rb1)
            pred = md.predict([[sepal_lenght,sepal_width,petal_length,petal_width]])
    return render_template("redirect.html",result=flw[pred[0]])
    


if __name__ == "__main__" :
    app.run(debug=True)
