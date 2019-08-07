# Crime-Classification
---
Code and Dataset for crime-classification.(Cooperation with Beijing Municipal Bureau of Public Security).  
## Overview  
This is a project that detect crimes from forum and online transaction. We use BOW, CNN, GRU, Attention to detect crimes, from very basic method to state-of-art method, so it is very **friendly** for nlp beginner.  
## Data 
The raw dataset is in **data/train/all_data.csv**. It has three columns, which are 
* id
* text
* domain 

The word embeddings can be trained by running **[code/tokenize.py](https://github.com/qhd1996/Crime-Classification-Chinese/blob/master/code/tokenize.py)**. The parameters are fine-tuned. The pre-trained word-embeddings will be save in folder **/data/models/**.  
## Training
Open Pycharm, under **code/model_name(bow, cnn, gru, attention)/** :
```
run model_name_train.py
```
If you want to change hyper-parameters, see **code/utils/utils.py/class Config**. 
After training, you can see pretrained model under **code/model_name(bow, cnn, gru, attention)/**.
## Test
```
run model_name_test.py
```
And you will get predictions. 
## Evaluation
After you get predictions, you can run **[code/evaluation/evaluation.py](https://github.com/qhd1996/Crime-Classification-Chinese/blob/master/code/evaluation/evaluation.py)** to get formated evaluation report.
We use accuracy, precision, recall, f1-score, support,micro avg, macro avg, weighted avg to evaluate these models. The evaluation file will be saved in  folder **/data/output/**.  
Roughly, BOW < GRU < CNN < Attention.  
## Dependencies
python 3.6
* jieba 0.39
* gensim 3.7.1
* pytorch 1.1.0(not 1.0.1, because of pack_padded_sequence)
* numpy 1.16.2  



