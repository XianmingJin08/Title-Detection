### Introduction:

I fine-tuned a Bert model in this project and concatenated the output from the Bert model with the other features together to do title detection. 

## Installation
- Clone this repo:
```bash
git clone https://github.com/XianmingJin08/Title-Detection.git
cd Title-Detection
```
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used is the provided dataset which is in the data/raw folder. I have preprocessed the data and stored them in the data/preprocessed using a script called preprocessing_dataset in the src/features. You can call it by:
```bash
python ./src/features/preprocessing_dataset.py
```
Overall, I firstly resolve an encoding issue when loading the dataset using pandas.read_csv function. Then, I removed the feature "FontType" as I found it is the same across all the datasets, which is redundant in this case. I then changed the Boolean values of features such as IsBold to 0s and 1s to be easier to feed to the classifier. Lastly, I transformed the location variables to zero means and unit variance, making the classifier's performance more stable. 

### 1) Training
There are two types of models you could train in this case: the simple Bert model or the BertSequenceClassifier model, which was pre-train on sequence classification tasks. Note an evaluation is done automatically after training.

For Bert model:
```bash
python main.py --path=[the folder you want to store or load your model] --training
```
For BertSequence model:
```bash
python main.py --bertSequence --path=[the folder you want to store or load your model] --training
```

### 2) Testing
I have trained two models and uploaded them onto google driver. You could download both models by running:
```bash
bash ./scripts/download_model.sh
```
This would create two folders: Bert and bertSequence, and download the models into the folders.
Similarly, you could define whether you want to predict using Bert or BertSequence by --bertSequence flat. One example using Bert would be:

```bash
python main.py --path=/models/bert 
```


  
