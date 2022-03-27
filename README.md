# Funded Loan Logistic Regression

A case is demonstrated where the amount of funded loans versus loans that will not fund.  First the amount of funded and loans that do not qualify are visualized then corrected to have more balanced training data.  

Header of data:

![image](https://user-images.githubusercontent.com/58529391/160301127-06015402-910d-486c-afdd-419b0732ebd6.png)

Data distribution pre-pre-processing:

![image](https://user-images.githubusercontent.com/58529391/160301156-12613ba3-43f8-4a4d-a11c-04b0a36e6343.png)

Data destribution after random choosing to have even distribution of classes:

![image](https://user-images.githubusercontent.com/58529391/160301957-295d5d10-02cc-430d-968f-e0626bf2972a.png)

Second the data is ran through a standard neural network to test performance. As there is only two classes of loans (funded vs. not funded) the data is then put through a logistic regression model to compare what model is better to use.

Lastly the roc and the area under the roc curve are compared to gague the logistic regression model performance.  
