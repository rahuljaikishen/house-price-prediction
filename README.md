# House prices prediction - a prodcutionized model

## About the code

The code is written to clean and preprocess data to train a linear regression model for house price prediction. Constants in the code can be updated to modify the features utilized to train. The code also includes an inference portion that uses the pre-trained model to make predictions.

### train.py
This file is used to train our model. We can change the model save path, and features utilized for training purposes.

Import and use 
```
from house_prices.train import build_model # def build_model(data: pd.DataFrame) -> dict[str, str]:
import pandas as pd

data = pd.read_csv('../PATH-TO-TRAIN-DATA')
score = build_model(data) #gets the rmsle score for the model trained

```

### inference.py
This file should be used on production to get the predictions for the house prices.

Import and use
```
from house_prices.inference import make_predictions # def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
import pandas as pd

data = pd.read_csv('../PATH-TO-LIVE-DATA')
predictions = make_predictions(data) #gets the predicted house prices

```

### preprocess.py
Methods in this file are used while training the model. Data cleaning, standardizing numerical features and encoding categorical features using OneHotEncoder can be found here.


## Project setup
  - conda create -y python=3.9 --name house_prices
  - conda activate house_prices
  - pip install -r requirements.txt
