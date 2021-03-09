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

### Preprocessed data

- data/disaster_categories.csv
- data/disaster_messages.csv

### Python Code

- data/process_data.py
- models/train_classifier.py
- app/run.py

### Processed data

- data/DisasterResponse.db
- models/classifier.pkl

### HTML

- app/templates/go.html
- app/templates/master.html


The full set of files used in this data analytic study including (a) my python code in jupyter format and (b) two .csv files of the dataset publicly available in Kaggle.  
The only jupyter notebook available here is to showcase the analytic work answering the questions as raised above regarding the International Student Survey. 
The literal analysis is posted in Medium.  This README also serves as a supplement for the readers who are interested in how I code into the data analysis.

## Results<a name="results"></a>

The main analytic findings can be found at the Medium post available [here](https://jimpikkin.medium.com/a-quick-glance-of-student-time-management-vs-performance-8b1815e2d5).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thanks Darwin Li for collecting the survey data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/xiaowenlimarketing/international-student-time-management).  Feel free to use the code here.

