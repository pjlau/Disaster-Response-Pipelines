# Disaster Response Pipelines

In this project, a supervised machine-learning is implemented with large labeled database. 

### Table of Contents

1. [Introduction](#introduction)
2. [Project Rationale](#rationale)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Introduction <a name="introduction"></a>

All libraries cited in the code are pre-installed along with the Anaconda distribution of Python. The code should be executable using Python versions 3.*.

## Project Rationale<a name="rationale"></a>

This project took a deeper look at a survey on a small international student body conducted in 2019. I was interested in knowing which demographic tend to do well/poorly and what can be improved in the course design to help them do better or be more interested. Thus, I raised a few questions and answered them by analyzing the survey data:

1. Which age group demonstrates the best academic performance?
2. Which program has the students tend to perform well?
3. Which survey question has the highest correlation to grade performance of the students?
4. Does attending the classes help the students get better grade?


## File Descriptions <a name="files"></a>

| Preprocessed data | Python Code | Processed data | HTML |
| --- | --- | --- | --- |
| data/disaster_categories.csv | data/process_data.py | data/DisasterResponse.db | app/templates/go.html |
| data/disaster_messages.csv | models/train_classifier.py | models/classifier.pkl | app/templates/master.html |
| | app/run.py | |


## Results<a name="results"></a>

The main analytic findings can be found at the Medium post available [here](https://jimpikkin.medium.com/a-quick-glance-of-student-time-management-vs-performance-8b1815e2d5).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thanks Darwin Li for collecting the survey data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/xiaowenlimarketing/international-student-time-management).  Feel free to use the code here.

