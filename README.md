# Remaining Useful Life predictions for electrical machines by physics-based machine learning

We introduce physics-informed neural netwroks for prognostics, which is the engineering discipline of predicting the time in which a system will fail. Centred on predicting the Remaining Useful Life through Health Index formulation, this paper presents a Physics-Based Machine Learning prognostics framework that imposes knowledge of the system's Health Index as loss functions. A case implementation on experimental information from 8 induction engines was conducted. This paper carefully assesses the performance of the physics-based machine learning model by comparing the predictions with available experimental data. The investigations reveal that the physics-based machine learning model can accurately improve the Remaining Useful Life predictions by more than 30 percent as compared to direct methods. Owing to the additional knowledge, less labelled training data is required, and the predictions are scientifically consistent with the known knowledge. The foray of physics-based machine learning to prognostics evidences the great potential of its application for broader applications in prognostics.  

In general, there are a series of steps taken to predicting the remaining useful life of a system. 

The first step is data collection and pre-processing where raw data is cleaned to ensure its accuracy, completeness and consistency. Raw sensor data were collected by Professor Wesley and his team from the University of Tennessee. They conducted thermal ageing on eight induction motors and collected the following 11 channels of raw signals.
- 3 phases of current (Current 1, 2, 3)​
- 3 phases of voltage (Voltage 1, 2, 3)​
- 2 accelerometer signal (Vibration 1, 2)​
- Tachometer signal (motor speed)​
- Thermocouple signal (temperature) ​
- Acoustic sensor signal (sound)



The second step is feature engineering which can be broken down into two sub steps, mainly (1) feature extraction which converts the pre-processed raw data into features which can explain the data better and (2) feature selection which ranks the features based on its distinctiveness and extracts the most important features that are relevant for the problem. 
The last step is to use the features to predict the remaining useful life. There are three main approaches to do so. The first approach is Data driven approach which uses statistical models and machine learning techniques to predict the RUL. The second approach is model based approach which employs physical models to predict the RUL And lastly, there is the hybrid approach which combines both data driven and model based approach to leverage on the strengths of both to enhance the accuracy of the RUL prediction.


