# Disaster-Response-App

The app is hosted on heroku but note that the model had to be downgraded to fit memory limits, so it isn't as accurate as it could have been.
https://disaster-response-figure8.herokuapp.com/

## Instructions
To deploy locally:
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
```                       
                        precision    recall  f1-score   support

               related       0.81      0.97      0.89      3991
               request       0.88      0.42      0.57       869
                 offer       0.00      0.00      0.00        24
           aid_related       0.76      0.63      0.69      2149
          medical_help       0.60      0.06      0.11       393
      medical_products       0.72      0.05      0.10       242
     search_and_rescue       0.83      0.04      0.07       132
              security       1.00      0.01      0.02       113
              military       0.75      0.05      0.10       177
                 water       0.86      0.29      0.43       302
                  food       0.87      0.46      0.60       549
               shelter       0.88      0.26      0.40       433
              clothing       0.83      0.06      0.11        84
                 money       0.80      0.03      0.07       117
        missing_people       0.00      0.00      0.00        70
              refugees       0.50      0.01      0.02       169
                 death       0.88      0.12      0.21       236
             other_aid       0.62      0.01      0.03       698
infrastructure_related       0.00      0.00      0.00       337
             transport       0.88      0.06      0.11       228
             buildings       0.83      0.06      0.11       256
           electricity       0.80      0.04      0.07       103
                 tools       0.00      0.00      0.00        35
             hospitals       0.00      0.00      0.00        53
                 shops       0.00      0.00      0.00        21
           aid_centers       0.00      0.00      0.00        62
  other_infrastructure       0.00      0.00      0.00       235
       weather_related       0.87      0.62      0.72      1497
                floods       0.95      0.36      0.52       450
                 storm       0.80      0.39      0.53       523
                  fire       0.50      0.02      0.04        44
            earthquake       0.91      0.76      0.83       491
                  cold       0.90      0.08      0.16       106
         other_weather       0.78      0.03      0.05       280
         direct_report       0.86      0.35      0.50       983

             micro avg       0.82      0.50      0.62     16452
             macro avg       0.62      0.18      0.23     16452
          weighted avg       0.77      0.50      0.54     16452
           samples avg       0.69      0.47      0.51     16452
```
