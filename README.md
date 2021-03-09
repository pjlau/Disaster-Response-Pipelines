# Disaster Response Pipelines

In this project, a supervised machine-learning is implemented with large labeled database. 

### Table of Contents

1. [Instruction](#instruction)
2. [Project Rationale](#rationale)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Instruction <a name="instruction"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `cd /home/workspace/app`
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Rationale<a name="rationale"></a>

This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety. 

1. Which age group demonstrates the best academic performance?
2. Which program has the students tend to perform well?
3. Which survey question has the highest correlation to grade performance of the students?
4. Does attending the classes help the students get better grade?


## File Descriptions <a name="files"></a>

| Preprocessed data | Python Code | Processed data | HTML |
| --- | --- | --- | --- |
| `data/disaster_categories.csv` | `data/process_data.py` | `data/DisasterResponse.db` | `app/templates/go.html` |
| `data/disaster_messages.csv` | `models/train_classifier.py` | `models/classifier.pkl` | `app/templates/master.html` |
| | `app/run.py` | |


## Results<a name="results"></a>

Some main analytic findings are presented as follows.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thanks Hugging Face for collecting the dataset of disaster response message.  You can find the data and other descriptive information at the GitHub link available [here](https://github.com/huggingface/datasets/tree/master/datasets/disaster_response_messages).  Feel free to use the code here.

