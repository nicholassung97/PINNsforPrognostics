# Remaining Useful Life predictions for electrical machines by physics-based machine learning
## Introduction
We introduce physics-informed neural netwroks for prognostics, which is the engineering discipline of predicting the time in which a system will fail. Centred on predicting the Remaining Useful Life through Health Index formulation, this paper presents a Physics-Based Machine Learning prognostics framework that imposes knowledge of the system's Health Index as loss functions. A case implementation on experimental information from 8 induction engines was conducted. This paper carefully assesses the performance of the physics-based machine learning model by comparing the predictions with available experimental data. The investigations reveal that the physics-based machine learning model can accurately improve the Remaining Useful Life predictions by more than 30 percent as compared to direct methods. Owing to the additional knowledge, less labelled training data is required, and the predictions are scientifically consistent with the known knowledge. The foray of physics-based machine learning to prognostics evidences the great potential of its application for broader applications in prognostics.  

Implementation using Tensorflow 2.0 can be accessed here:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/nicholassung97/PINNsforPrognostics/blob/main/tensorflow2.0.ipynb)

Implementation using Tensorflow 1.0 can be accessed here:
[<img src="data:image/jpeg;base64,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" width="24">](https://github.com/nicholassung97/PINNsforPrognostics/blob/main/tensorflow1.0.py)

Full Final Year Report can be accessed here:
[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/PDF_file_icon.svg/1667px-PDF_file_icon.svg.png" width="24">](https://github.com/nicholassung97/PINNsforPrognostics/blob/main/FYP_B062_NicholasSung.pdf)


## Methodology
In general, there are a series of steps taken to predicting the remaining useful life of a system. 

### Data Collection and Pre-processing
The first step is data collection and pre-processing where raw data is cleaned to ensure its accuracy, completeness and consistency. Raw sensor data were collected by Professor Wesley and his team from the University of Tennessee. They conducted thermal ageing on eight induction motors and collected the following 11 channels of raw signals.
- 3 phases of current (Current 1, 2, 3)​
- 3 phases of voltage (Voltage 1, 2, 3)​
- 2 accelerometer signal (Vibration 1, 2)​
- Tachometer signal (motor speed)​
- Thermocouple signal (temperature) ​
- Acoustic sensor signal (sound)

### Feature Engineering (Feature Extraction and Feature Selection)
The second step is feature engineering which can be broken down into two sub steps, mainly (1) feature extraction which converts the pre-processed raw data into features which can explain the data better and (2) feature selection which ranks the features based on its distinctiveness and extracts the most important features that are relevant for the problem. 
The pre-processing and feature extraction stages were conducted by Yang Feng and his team by substituting the missing values with values from the previous cycle and applying 11 statistical indicators as shown in the table to each of the 11 channels of raw data to obtain 121 features.

<img src="https://user-images.githubusercontent.com/84385004/192479900-efd1c076-d5c1-4460-b5cd-f18edb63cee8.png" width="500" height="400" />


Having the features created, I proceeded to do feature selection to rank 121 features based on its distinctiveness and extracted the 20 most important features. The method used was fishers ratio which measures the discriminating power between a healthy class and an unhealthy class. We assumed that the motor was healthy in the first four cycles and unhealthy in the last four cycles. Essentially, the larger the ratio, the more the feature has changed when the motor degrades from a healthy to an unhealthy state, which makes the feature more distinctive.

<img src="https://user-images.githubusercontent.com/84385004/192479845-c7a7d41c-3dab-47ad-a32e-a6066c7d473a.png" width="400" height="200" />

### Feature to RUL prediction
The last step is to use the features to predict the remaining useful life. 
In this project, prediction was achieved through a two-stage modelling process with health index as an intermediary. The health index represents the state of health of the system. 
In the first stage, the feature will be used as input into the physics-based machine learning framework to predict the Health Index (HI) of the system. This framework will be the main contribution of this project and it was build using tensorflow 2.0. In the second stage, the health index is mapped to the predicted RUL, after which the final RUL prediction is produced through an ensemble approach.

<img src="https://user-images.githubusercontent.com/84385004/192479684-db6e33b7-b2c9-4136-9b6a-7fcf47e7a096.png" width="450" height="175" />

### Physics-Based Machine Learning framework
This proposed physics-based machine learning model will incorporate knowledge through the following 3 rules of the health index:
1. Predicted Health Index value for the subsequent cycle cannot increase
2. Deviation of the predicted Health Index value within each cycle should be small 
3. Drop in the predicted Health Index for the subsequent cycle is small.

#### Rules will be imposed as penalty terms in the loss function
#### Rule 1
The first rule is that the predicted health index for subsequent cycles cannot increase (this means that the health of the system cannot improve overtime)
To enforce this rule into the neural network we add Loss Rule 1 as a penalty term into the loss function. How this loss term works is if the mean health index for cycle i+1 is more than the mean health index for cycle i, this indicates that the health index increases and the term in the bracket will be positive. 

When passed through the ReLu function, the output will be reflected linearly, and the optimisation process will attempt to reduce this term until it is close to zero. When it does so, the mean predicted health index does not increase anymore. 
In the event that the mean health index already decreases, the term in the ReLu function will be negative and the loss term will be zero, hence physical consistency is achieved.

<img src="https://user-images.githubusercontent.com/84385004/192483106-df011385-9b84-4f8f-806c-fffb0780e3be.png" width="500" height="100" />

#### Rule 2
The second rule is that all the predicted health index for the same cycle should be similar. 
To enforce this rule into the neural network we add Loss Rule 2 as a penalty term into the loss function
How this loss term works is that it is equal to the mean absolute percentage error between the health index predictions within a cycle and their average value.
Essentially, a larger mean absolute percentage error will reflect a larger deviation of the health index within the cycle.
The optimisation process will attempt to reduce this term until it is close to zero hence forcing this physical constraint of a small deviation within each cycle. 

<img src="https://user-images.githubusercontent.com/84385004/192485136-fb2dcdb1-a9e9-40b2-9c9f-41e5be41d9ab.png" width="400" height="80" />

Where t represents the data number within a cycle, n represents the last data set in the cycle.

#### Rule 3
For the last rule, we want to limit the drop in the health index by 20 percent. 
In other words, from one cycle to the next, we do not expect the health of the system to degrade too drastically.
As seen from the inequality formula, if the drop in the mean health index is more than 20 percent, we can bring over the right-hand side term to obtain the following. 
If this term is positive, it means that the drop is more than 20 percent and when we pass it through the ReLu function, the output will be positive too. The optimisation process will attempt to reduce this term until it is close to zero. When it does so, the mean predicted health index should not decrease by more than 20 percent. 

<img src="https://user-images.githubusercontent.com/84385004/192485962-78e78bb8-f488-41f5-b0a5-23771b3bd47f.png" width="450" height="50" />

<img src="https://user-images.githubusercontent.com/84385004/192486112-c9870609-ded5-4765-b509-3c2271eee2ef.png" width="450" height="100" />

#### Boundary Conditions
Moving on to the boundary conditions, we take the first cycle to be healthy, hence it will have a health index of 100% and the last cycle to be unhealthy, hence it will have a health index of 10%. The loss term will be mean square error between the predicted value and true value for the first and last cycle.

<img src="https://user-images.githubusercontent.com/84385004/192488207-bdc49ec0-b094-42d4-bf1c-5104309edc95.png" width="600" height="100" />

#### Overall Physics Based Machine Learning Framework
Ultimately, we can obtain our overall loss function by adding all the loss terms as shown in this framework. 
The 20 most distinctive features and cycles are input to predict the health index which should then be constrained by the three rules and the boundary conditions

![image](https://user-images.githubusercontent.com/84385004/192488773-d681e9bc-3252-45ff-9ef6-7d0712d0c872.png)


#### Ensemble Approach: Train and Tested with same engine
Using this physics based machine learning framework, we used the 20 features and cycles from each of the 8 engines to train 8 separate health index prediction models. When we ran the features of each engine through their corresponding model. We found that all the engines show a similar degradation trend which illustrates (1) a gradual decrease in health index, (2) the small deviation in Health Index for the same cycle and (3) the boundary conditions are met  where the first cycle is 100% healthy and last cycle is 10% healthy. So essentially, ALL the rules and boundary conditions are abided by.

![image](https://user-images.githubusercontent.com/84385004/192489388-7078a39e-6882-40ed-b8f8-b168e8fda08f.png)

#### Health Index to RUL Prediction
With the 8 models trained, the features from one test motor will run through the prediction models trained by the remaining 7 motors to obtain 7 predicted health index. This Health index is then mapped to RUL through interpolation or extrapolation.

![image](https://user-images.githubusercontent.com/84385004/192489928-b679fcae-8569-41f8-81db-7a2b9084f7db.png)

## Results
To gauge the performance of this method, we utilise two error metrics and compare them with the direct method. The direct method maps features directly to the RUL during training. The first error metric is root mean square error which measures the differences between the true RUL and the predicted RUL for every cycle
What we observe is that RUL prediction using physics-based machine learning out-performs the direct method for most engines and on average, reduces the root mean square error by 31.5%.

![image](https://user-images.githubusercontent.com/84385004/192490645-fb844a15-8ca0-4b6a-b38e-132aa7d4f629.png)

The second error metric used was the mean absolute percentage error which is a measure of prediction accuracy. 

Similar to root mean square error, the RUL prediction using physics-based machine learning out-performs the direct method for most engines and on average, reduces the mean absolute percentage error by 33.6%.

![image](https://user-images.githubusercontent.com/84385004/192490838-e10ffa5e-0bd8-463d-8865-32abc4c54e4f.png)

