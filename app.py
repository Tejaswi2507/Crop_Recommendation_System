from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained models and their accuracies
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))
decision_tree_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
model_accuracies = pickle.load(open('model_accuracies.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    input_values = [
        float(request.form['nitrogen']),
        float(request.form['phosphorous']),
        float(request.form['potassium']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]

    # Make predictions using all models
    rf_recommended_crop = random_forest_model.predict([input_values])[0]
    dt_recommended_crop = decision_tree_model.predict([input_values])[0]
    svm_recommended_crop = svm_model.predict([input_values])[0]

    # Determine the best recommendation based on accuracy
    if model_accuracies['random_forest'] >= model_accuracies['decision_tree'] and model_accuracies['random_forest'] >= model_accuracies['svm']:
        best_recommended_crop = rf_recommended_crop
        best_model = "Random Forest"
    elif model_accuracies['decision_tree'] >= model_accuracies['svm']:
        best_recommended_crop = dt_recommended_crop
        best_model = "Decision Tree"
    else:
        best_recommended_crop = svm_recommended_crop
        best_model = "SVM"

    return render_template('result.html', 
                           rf_crop=rf_recommended_crop, 
                           dt_crop=dt_recommended_crop, 
                           svm_crop=svm_recommended_crop, 
                           rf_accuracy=model_accuracies['random_forest'], 
                           dt_accuracy=model_accuracies['decision_tree'], 
                           svm_accuracy=model_accuracies['svm'], 
                           best_crop=best_recommended_crop,
                           best_model=best_model)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
