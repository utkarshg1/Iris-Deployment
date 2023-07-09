# Creating Flask web application
from flask import Flask,request, render_template
import pickle
import pandas as pd

# Create the Flask app
application = Flask(__name__)
app = application

# Create Homepage path
@app.route('/')
def home_page():
    return render_template('index.html')

# Creating prediction code
@app.route('/predict',methods=['POST'])
def predict_point():
    if request.method=='GET':
        render_template('index.html')
    else:
        # Load the le pickle files
        with open('notebook/LabelEnc.pkl','rb') as file1:
            le = pickle.load(file1)
        # Load the Scaler file
        with open('notebook/Scaler.pkl','rb') as file2:
            scaler = pickle.load(file2)
        # Load the model
        with open('notebook/model.pkl','rb') as file3:
            model = pickle.load(file3)
        # Read all the values
        sep_len = float(request.form.get('sepal_length'))
        sep_wid = float(request.form.get('sepal_width'))
        pet_len = float(request.form.get('petal_length'))
        pet_wid = float(request.form.get('petal_width'))
        # Create the dataframe for above
        xnew = pd.DataFrame([sep_len,sep_wid,pet_len,pet_wid]).T
        xnew.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        # Preprocess the dataframe
        xnew_pre = pd.DataFrame(scaler.transform(xnew),columns=xnew.columns)
        # Predict the data
        pred = model.predict(xnew_pre)
        pred_final = le.inverse_transform(pred)[0]
        # Probability of pred
        prob = model.predict_proba(xnew_pre).max()
        # Prediction string
        prediction = f'{pred_final} with Probability : {prob:.4f}'
        
    return render_template('index.html',prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)