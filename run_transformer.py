import pandas as pd
from preprocess import PreProcess

path = 'D:/Python_scripts_test/Transformers/'
data=pd.read_csv(path + 'data.csv')
fix_seq_size=60
preprocesor = PreProcess(data, 'english', 'spanish', fix_seq_size)
X_train, X_test, Y_train, Y_test = preprocesor.split_data()
eng_train, eng_test, eng_vocab, eng_ivocab = preprocesor.data_tokenization(X_train, X_test)
esp_train, esp_test, esp_vocab, esp_ivocab = preprocesor.data_tokenization(Y_train, Y_test)

