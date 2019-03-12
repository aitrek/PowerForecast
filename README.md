# Project

## Name

* Power Forecast

## Description

* Power forecast model based on Machine Learning and Neural Networks algorithm
    + Machine Learning algorithm include:
        + Ridge Regression/Lasso Regression/SVM/XGBoost
    + Neural Networks algorithm include:
        + MLP/LSTM
        
* Field name description:
    + y -> Actual power generation in the month
    + x1 -> Shanghai electricity consumption
    + x2 -> Shanghai received electricity(mouth)
    + x3 -> Shanghai power generation
    + x41 -> The number of days with a temperature less than 4 degrees Celsius
    + x42 -> The number of days at 5-14 degrees Celsius
    + x43 -> The number of days at 15-24 degrees Celsius
    + x44 -> The number of days at 25-34 degrees Celsius
    + x45 -> The number of days at 35-40 degrees Celsius
    + x51 -> The number of days of the rest day of the month
    + x52 -> The number of days of the working day of the month
    + x6 -> Does this month include the Spring Festival? 1: yes, 0: no
    + x7 -> Urban and rural residents' electricity consumption
    + x8 -> Shanghai primary industry electricity consumption
    + x9 -> Shanghai secondary industry electricity consumption
    + x10 -> Shanghai tertiary industry electricity consumption
    + x11 -> Shanghai CPI for the month
    + x12 -> Industrial producer ex-factory price index

## Note

* Data has been desensitized

## Dependent python library

```
Keras==2.1.2
matplotlib==2.2.3
numpy==1.13.3
pandas==0.21.0
scikit-learn==0.20.2
statsmodels==0.9.0
tensorflow==1.4.1
typing==3.5.3.0
xgboost==0.81
```

## Command

```shell
pip3 freeze > requirements.txt
------
create a github project
git init
git config user.name HjwGivenLyy
git config user.email 1752929469@qq.com
git remote add origin git@github.com:htpauleta/PowerForecast.git
git add .
git commit -m "init a project"
git push -u origin master
```
mongodb:
  host: "144.202.41.144"
  port: 27017
  pwd: "pauletapass"
  user: "pauleta"
  db: "xscore"