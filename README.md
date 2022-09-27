# Remaining Useful Life predictions for electrical machines by physics-based machine learning
## Introduction
We introduce physics-informed neural netwroks for prognostics, which is the engineering discipline of predicting the time in which a system will fail. Centred on predicting the Remaining Useful Life through Health Index formulation, this paper presents a Physics-Based Machine Learning prognostics framework that imposes knowledge of the system's Health Index as loss functions. A case implementation on experimental information from 8 induction engines was conducted. This paper carefully assesses the performance of the physics-based machine learning model by comparing the predictions with available experimental data. The investigations reveal that the physics-based machine learning model can accurately improve the Remaining Useful Life predictions by more than 30 percent as compared to direct methods. Owing to the additional knowledge, less labelled training data is required, and the predictions are scientifically consistent with the known knowledge. The foray of physics-based machine learning to prognostics evidences the great potential of its application for broader applications in prognostics.  

## Methodology
In general, there are a series of steps taken to predicting the remaining useful life of a system. 

The first step is data collection and pre-processing where raw data is cleaned to ensure its accuracy, completeness and consistency. Raw sensor data were collected by Professor Wesley and his team from the University of Tennessee. They conducted thermal ageing on eight induction motors and collected the following 11 channels of raw signals.
- 3 phases of current (Current 1, 2, 3)​
- 3 phases of voltage (Voltage 1, 2, 3)​
- 2 accelerometer signal (Vibration 1, 2)​
- Tachometer signal (motor speed)​
- Thermocouple signal (temperature) ​
- Acoustic sensor signal (sound)

The second step is feature engineering which can be broken down into two sub steps, mainly (1) feature extraction which converts the pre-processed raw data into features which can explain the data better and (2) feature selection which ranks the features based on its distinctiveness and extracts the most important features that are relevant for the problem. 
The pre-processing and feature extraction stages were conducted by Yang Feng and his team by substituting the missing values with values from the previous cycle and applying 11 statistical indicators as shown in the table to each of the 11 channels of raw data to obtain 121 features.

<img src="https://user-images.githubusercontent.com/84385004/192479900-efd1c076-d5c1-4460-b5cd-f18edb63cee8.png" width="500" height="400" />


Having the features created, I proceeded to do feature selection to rank 121 features based on its distinctiveness and extracted the 20 most important features. The method used was fishers ratio which measures the discriminating power between a healthy class and an unhealthy class. We assumed that the motor was healthy in the first four cycles and unhealthy in the last four cycles. Essentially, the larger the ratio, the more the feature has changed when the motor degrades from a healthy to an unhealthy state, which makes the feature more distinctive.

<img src="https://user-images.githubusercontent.com/84385004/192479845-c7a7d41c-3dab-47ad-a32e-a6066c7d473a.png" width="600" height="300" />


The last step is to use the features to predict the remaining useful life. 
In this project, prediction was achieved through a two-stage modelling process with health index as an intermediary. The health index represents the state of health of the system. 
In the first stage, the feature will be used as input into the physics-based machine learning framework to predict the Health Index (HI) of the system. This framework will be the main contribution of this project and it was build using tensorflow 2.0. In the second stage, the health index is mapped to the predicted RUL, after which the final RUL prediction is produced through an ensemble approach.

<img src="https://user-images.githubusercontent.com/84385004/192479684-db6e33b7-b2c9-4136-9b6a-7fcf47e7a096.png" width="900" height="400" />

### Physics-Based Machine Learning framework
This proposed physics-based machine learning model will incorporate knowledge through the following 3 rules of the health index:
1. Predicted Health Index value for the subsequent cycle cannot increase
2. Deviation of the predicted Health Index value within each cycle should be small 
3. Drop in the predicted Health Index for the subsequent cycle is small.

#### Rules will be imposed as penalty terms in the loss function
The first rule is that the predicted health index for subsequent cycles cannot increase (this means that the health of the system cannot improve overtime)
To enforce this rule into the neural network we add Loss Rule 1 as a penalty term into the loss function. How this loss term works is if the mean health index for cycle i+1 is more than the mean health index for cycle i, this indicates that the health index increases and the term in the bracket will be positive. 

When passed through the ReLu function, the output will be reflected linearly, and the optimisation process will attempt to reduce this term until it is close to zero. When it does so, the mean predicted health index does not increase anymore. 
In the event that the mean health index already decreases, the term in the ReLu function will be negative and the loss term will be zero, hence physical consistency is achieved.

<img src="https://user-images.githubusercontent.com/84385004/192483106-df011385-9b84-4f8f-806c-fffb0780e3be.png" width="500" height="100" />

The second rule is that all the predicted health index for the same cycle should be similar. 
To enforce this rule into the neural network we add Loss Rule 2 as a penalty term into the loss function
How this loss term works is that it is equal to the mean absolute percentage error between the health index predictions within a cycle and their average value.
Essentially, a larger mean absolute percentage error will reflect a larger deviation of the health index within the cycle.
The optimisation process will attempt to reduce this term until it is close to zero hence forcing this physical constraint of a small deviation within each cycle. 

<img src="https://user-images.githubusercontent.com/84385004/192485136-fb2dcdb1-a9e9-40b2-9c9f-41e5be41d9ab.png" width="450" height="100" />


Where t represents the data number within a cycle, n represents the last data set in the cycle.

