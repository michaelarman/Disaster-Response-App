# Disaster-Response-App

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.
    - To run the ETL pipeline that cleans data and stores in database
  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains the classifier and saves the model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to the link provided in the terminal


## Project Overview
In this project, we will analyze disaster [data from Figure Eight](https://appen.com/datasets/combined-disaster-response-data/) to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. 
We will be creating a machine learning pipeline to categorize these events so that one can send the messages to an appropriate disaster relief agency.

Running the code will output a web app where an emergency worker can input a new message and get classification results in several categories.

## Project Components
There are three main components to this project 
1. ETL Pipeline
[data/process_data.py](https://github.com/michaelarman/Disaster-Response-App/blob/master/data/process_data.py) contains a data cleaning pipeline that:
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database

2. ML Pipeline
[models/train_classifier.py](https://github.com/michaelarman/Disaster-Response-App/blob/master/models/train_classifier.py) contains a machine learning pipeline that:

    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file
    
3. Flask Web App
For displaying all results and and using the model to classify user input queries.

## Project Discussion
The model created used a MultiOutput Random Forest Classifier which worked well. I compared it to other classifiers such as KNN but Random Forest was more accurate and also faster.
The model isn't ideal for predicting future disasters because it needs more data to be trained on.

The evaluation of the model can be seen below where each segment is it's own category (there are 36 in total)
```            precision    recall  f1-score   support

           0       0.72      0.28      0.40      1230
           1       0.81      0.96      0.88      3981
           2       0.52      0.36      0.43        33

    accuracy                           0.80      5244
   macro avg       0.68      0.54      0.57      5244
weighted avg       0.79      0.80      0.76      5244

              precision    recall  f1-score   support

           0       0.89      0.99      0.94      4371
           1       0.88      0.40      0.55       873

    accuracy                           0.89      5244
   macro avg       0.89      0.70      0.75      5244
weighted avg       0.89      0.89      0.87      5244

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      5227
           1       0.00      0.00      0.00        17

    accuracy                           1.00      5244
   macro avg       0.50      0.50      0.50      5244
weighted avg       0.99      1.00      1.00      5244

              precision    recall  f1-score   support

           0       0.78      0.87      0.82      3127
           1       0.77      0.64      0.70      2117

    accuracy                           0.78      5244
   macro avg       0.78      0.76      0.76      5244
weighted avg       0.78      0.78      0.77      5244

              precision    recall  f1-score   support

           0       0.93      1.00      0.96      4849
           1       0.60      0.05      0.08       395

    accuracy                           0.93      5244
   macro avg       0.76      0.52      0.52      5244
weighted avg       0.90      0.93      0.90      5244

              precision    recall  f1-score   support

           0       0.96      1.00      0.98      4993
           1       0.79      0.08      0.14       251

    accuracy                           0.95      5244
   macro avg       0.87      0.54      0.56      5244
weighted avg       0.95      0.95      0.94      5244

              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5116
           1       0.78      0.05      0.10       128

    accuracy                           0.98      5244
   macro avg       0.88      0.53      0.55      5244
weighted avg       0.97      0.98      0.97      5244

              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5134
           1       1.00      0.01      0.02       110

    accuracy                           0.98      5244
   macro avg       0.99      0.50      0.50      5244
weighted avg       0.98      0.98      0.97      5244

              precision    recall  f1-score   support

           0       0.97      1.00      0.98      5062
           1       0.83      0.03      0.05       182

    accuracy                           0.97      5244
   macro avg       0.90      0.51      0.52      5244
weighted avg       0.96      0.97      0.95      5244

              precision    recall  f1-score   support

           0       0.95      1.00      0.98      4911
           1       0.91      0.28      0.42       333

    accuracy                           0.95      5244
   macro avg       0.93      0.64      0.70      5244
weighted avg       0.95      0.95      0.94      5244

              precision    recall  f1-score   support

           0       0.94      0.99      0.96      4659
           1       0.86      0.46      0.60       585

    accuracy                           0.93      5244
   macro avg       0.90      0.73      0.78      5244
weighted avg       0.93      0.93      0.92      5244

              precision    recall  f1-score   support

           0       0.93      1.00      0.96      4772
           1       0.91      0.25      0.39       472

    accuracy                           0.93      5244
   macro avg       0.92      0.62      0.68      5244
weighted avg       0.93      0.93      0.91      5244

              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5163
           1       0.71      0.06      0.11        81

    accuracy                           0.99      5244
   macro avg       0.85      0.53      0.55      5244
weighted avg       0.98      0.99      0.98      5244

              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5129
           1       0.50      0.01      0.02       115

    accuracy                           0.98      5244
   macro avg       0.74      0.50      0.50      5244
weighted avg       0.97      0.98      0.97      5244

              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5186
           1       1.00      0.02      0.03        58

    accuracy                           0.99      5244
   macro avg       0.99      0.51      0.51      5244
weighted avg       0.99      0.99      0.98      5244

              precision    recall  f1-score   support

           0       0.96      1.00      0.98      5052
           1       0.40      0.01      0.02       192

    accuracy                           0.96      5244
   macro avg       0.68      0.50      0.50      5244
weighted avg       0.94      0.96      0.95      5244

              precision    recall  f1-score   support

           0       0.96      1.00      0.98      5013
           1       0.90      0.12      0.21       231

    accuracy                           0.96      5244
   macro avg       0.93      0.56      0.59      5244
weighted avg       0.96      0.96      0.95      5244

              precision    recall  f1-score   support

           0       0.87      1.00      0.93      4566
           1       0.82      0.01      0.03       678

    accuracy                           0.87      5244
   macro avg       0.85      0.51      0.48      5244
weighted avg       0.87      0.87      0.81      5244

              precision    recall  f1-score   support

           0       0.94      1.00      0.97      4938
           1       0.00      0.00      0.00       306

    accuracy                           0.94      5244
   macro avg       0.47      0.50      0.48      5244
weighted avg       0.89      0.94      0.91      5244

              precision    recall  f1-score   support

           0       0.96      1.00      0.98      5019
           1       0.88      0.06      0.12       225

    accuracy                           0.96      5244
   macro avg       0.92      0.53      0.55      5244
weighted avg       0.96      0.96      0.94      5244

              precision    recall  f1-score   support

           0       0.95      1.00      0.97      4964
           1       0.82      0.05      0.09       280

    accuracy                           0.95      5244
   macro avg       0.89      0.52      0.53      5244
weighted avg       0.94      0.95      0.93      5244

              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5146
           1       0.75      0.03      0.06        98

    accuracy                           0.98      5244
   macro avg       0.87      0.52      0.52      5244
weighted avg       0.98      0.98      0.97      5244

              precision    recall  f1-score   support

           0       0.99      1.00      1.00      5211
           1       0.00      0.00      0.00        33

    accuracy                           0.99      5244
   macro avg       0.50      0.50      0.50      5244
weighted avg       0.99      0.99      0.99      5244

              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5189
           1       0.00      0.00      0.00        55

    accuracy                           0.99      5244
   macro avg       0.49      0.50      0.50      5244
weighted avg       0.98      0.99      0.98      5244

              precision    recall  f1-score   support

           0       0.99      1.00      1.00      5211
           1       0.00      0.00      0.00        33

    accuracy                           0.99      5244
   macro avg       0.50      0.50      0.50      5244
weighted avg       0.99      0.99      0.99      5244

              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5190
           1       0.00      0.00      0.00        54

    accuracy                           0.99      5244
   macro avg       0.49      0.50      0.50      5244
weighted avg       0.98      0.99      0.98      5244

              precision    recall  f1-score   support

           0       0.96      1.00      0.98      5041
           1       0.00      0.00      0.00       203

    accuracy                           0.96      5244
   macro avg       0.48      0.50      0.49      5244
weighted avg       0.92      0.96      0.94      5244

              precision    recall  f1-score   support

           0       0.87      0.96      0.91      3812
           1       0.86      0.62      0.72      1432

    accuracy                           0.87      5244
   macro avg       0.87      0.79      0.82      5244
weighted avg       0.87      0.87      0.86      5244

              precision    recall  f1-score   support

           0       0.95      0.99      0.97      4828
           1       0.86      0.38      0.53       416

    accuracy                           0.95      5244
   macro avg       0.91      0.69      0.75      5244
weighted avg       0.94      0.95      0.94      5244

              precision    recall  f1-score   support

           0       0.95      0.99      0.97      4773
           1       0.78      0.45      0.57       471

    accuracy                           0.94      5244
   macro avg       0.86      0.72      0.77      5244
weighted avg       0.93      0.94      0.93      5244

              precision    recall  f1-score   support

           0       0.99      1.00      0.99      5181
           1       0.00      0.00      0.00        63

    accuracy                           0.99      5244
   macro avg       0.49      0.50      0.50      5244
weighted avg       0.98      0.99      0.98      5244

              precision    recall  f1-score   support

           0       0.98      0.99      0.98      4766
           1       0.89      0.75      0.82       478

    accuracy                           0.97      5244
   macro avg       0.93      0.87      0.90      5244
weighted avg       0.97      0.97      0.97      5244

              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5140
           1       0.86      0.06      0.11       104

    accuracy                           0.98      5244
   macro avg       0.92      0.53      0.55      5244
weighted avg       0.98      0.98      0.97      5244

              precision    recall  f1-score   support

           0       0.95      1.00      0.97      4966
           1       0.50      0.02      0.04       278

    accuracy                           0.95      5244
   macro avg       0.72      0.51      0.51      5244
weighted avg       0.92      0.95      0.92      5244

              precision    recall  f1-score   support

           0       0.87      0.99      0.92      4252
           1       0.86      0.34      0.49       992

    accuracy                           0.87      5244
   macro avg       0.86      0.67      0.71      5244
weighted avg       0.87      0.87      0.84      5244
