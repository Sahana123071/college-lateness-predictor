# College Lateness Predictor

A simple machine learning classifier built for an AI/ML internship task.

#Code
# minutes before 9:00 AM that you left the room
minutes_before_class = [5, 10, 15, 20, 25, 30, 8, 12, 35, 18, 22, 40, 6, 28]

# was I late that day?
result = ["Late","Late","Late","On Time","On Time","On Time","Late","Late",
          "On Time","Late","On Time","On Time","Late","On Time"]
import pandas as pd
df = pd.DataFrame({"minutes_before": minutes_before_class, "result": result})
df
from sklearn.linear_model import LogisticRegression

X = df[["minutes_before"]]
y = df["result"]

model = LogisticRegression()
model.fit(X, y)
model.predict([[15]])   # "if I leave 15 min before class, will I be late?"
for m in range(0, 45, 5):
    print(m, "min before ->", model.predict([[m]])[0])
## What it does
Predicts whether I'll be late to college based on how many minutes 
before class I leave my room, using data from my own daily routine.

## Why I picked this dataset
It's something from my real life — I could immediately sanity-check 
whether the model's predictions matched my own experience.

## What I found
The model learned that my personal tipping point is around 15-20 
minutes before class — leaving less than that, I'm consistently late; 
leaving 20+ minutes early, I'm on time. [Add here whether this matched 
what you expected, or surprised you.]

## Tools used
- Python
- pandas
- scikit-learn (Logistic Regression)
- Google Colab

## How to run
Open the `.ipynb` notebook in Google Colab and run the cells in order.
