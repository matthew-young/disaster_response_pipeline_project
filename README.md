Introduction

This project analyses the data on messages from disaster areas. Using thirty 35 categories to determine what kind of message someone has sent and how they can be aided. Using a webpage, you can enter your own message and try the program.

How to:
    Run the following commands in the project's root directory to set up your database and model.
        To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

    Run the following command in the app's directory to run your web app. python run.py

    Go to http://0.0.0.0:3001/


Thank you to Figure Eight and Udacity for the course material