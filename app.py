import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
loaded = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = loaded.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index.html',prediction_text='Possibility Of Heart Attack In 10 years is->{}'.format(output))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

