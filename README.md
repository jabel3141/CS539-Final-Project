Jason Abel  
Megan Deloach
Jack Gonsalves
Tai Zhou

https://jabel3141.github.io/CS539-Final-Project/



Github Repo:
https://github.com/jabel3141/CS539-Final-Project


This Project shows a redesigned website for the Billiards University Exam. It allows people to take both parts of the exam and see their statistics based on how they do.


## Project Description
The goal of this project was to analyze clickstream data from the PBS KIDS Measure Up! app games to predict how well they would do on some assessment in the app. The data is provided by PBS for a competition on Kaggle. PBS has always been known for helping young children learn everyday basic skills such as reading, writing, counting, and math. PBS is trying to get people to participate in the competition so that they can gain insights on what games help children learn best. In the end we were able to create one model that could predict better than simple models or random guessing, but not well enough to actually compete in the competition. 


## Project Data

The project data can be found at this kaggle webpage: https://www.kaggle.com/c/data-science-bowl-2019/data
The data in this repo is a small sample of the overall data.

The training data given to us was split between two files, one that was going to look exactly like the test file, and another containing the labels and some more information. The information combining the two files was the installation_id. In the training labels folder there were thousands of rows containing an instalation_id, game_session and some information on how well they did on that given assessment. PBS wanted us to predict the accuracy group of the last assessment for each installation_id; this is why multiple rows can share the same installation_id in the training_labels file. Below are tables representing each file with the first 100 entries of data.

In the training_labels file there is a lot of data that was given that is not given in the test file. This includes the number of correct answers, incorrect answers, accuracy, and the title of the assignment. This data could have possibly been used if we had thought a regression model on accuracy would be better, but we only cared about the accuracy group so we threw out all that other information as we could not use it on the test data. 

The training and test files contained almost exactly the same data. They contained the same columns, but held different information in the rows. The train and test file contained clickstream data, so each row in the csv contained information on action that the player made in the game. The data we eventually decided to use in each action was the game_session, time_stamp, installation_id, event_count, event_code, type, and world. Game_session and installation ids were the identifiers, time_stamps were used to organize data into training sets and the rest of the information was used for predictions and training. Type is the type of thing the person was playing on the app like an activity or game, event_count was which number action it was that the player made, event_code was something that described what the user did, and world was one of the major sections the user was playing in. The only difference between the test and training file was the fact the the test file did contained only the start code for the last assessment the user took, while training contained all the clickstream information for each assessment. 


## Project Models:
We used multiple models. How to download and run each model individually are on the website under each model

