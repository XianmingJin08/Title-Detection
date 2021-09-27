import chardet
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    with open('./data/raw/train_sections_data.csv', 'rb') as f:
        result_train = chardet.detect(f.read())  # or readline if the file is large
    with open('./data/raw/test_sections_data.csv', 'rb') as f:
        result_test = chardet.detect(f.read())  # or readline if the file is large
    train_full = pd.read_csv('./data/raw/train_sections_data.csv', encoding=result_train['encoding'])
    test_full = pd.read_csv('./data/raw/test_sections_data.csv', encoding=result_test['encoding'])
    train_full_cleaned = train_full.drop(['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'FontType'], axis=1)
    test_full_cleaned = test_full.drop(['FontType'], axis=1)
    train_full_cleaned[["IsBold", "IsItalic", "IsUnderlined"]] = train_full_cleaned[
        ["IsBold", "IsItalic", "IsUnderlined"]].astype(int)
    test_full_cleaned[["IsBold", "IsItalic", "IsUnderlined"]] = test_full_cleaned[
        ["IsBold", "IsItalic", "IsUnderlined"]].astype(int)
    train_cleaned, valid_cleaned = train_test_split(train_full_cleaned, test_size=0.1)
    scaler = preprocessing.StandardScaler().fit(train_cleaned.iloc[:, 4:8])
    train_cleaned.iloc[:, 4:8] = scaler.transform(train_cleaned.iloc[:, 4:8])
    valid_cleaned.iloc[:, 4:8] = scaler.transform(valid_cleaned.iloc[:, 4:8])
    test_full_cleaned.iloc[:, 4:8] = scaler.transform(test_full_cleaned.iloc[:, 4:8])
    train_cleaned.to_csv("./data/processed/train_cleaned.csv", index=False)
    valid_cleaned.to_csv("./data/processed/valid_cleaned.csv", index=False)
    test_full_cleaned.to_csv("./data/processed/test_cleaned.csv", index=False)