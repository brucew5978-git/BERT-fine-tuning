#Source: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('%d GPU(s) available' % torch.cuda.device_count())
    print('Using: ', torch.cuda.get_device_name(0))

else: 
    print("Using CPU... ")
    device = torch.device("cpu")

#note: need hugging face library to be installed (pip install transformers)

#using "Corpus of Linguistic Acceptability (CoLA)" dataset for sentence classification. Dataset contains labels for grammatically correct and incorrect datasets

import wget
import os

'''
print("Downloading dataset...")
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

if not os.path.exists('./cola_public_1.1.zip'):
    wget.download(url, './cola_public_1.1.zip')
'''
#for use on colab
#if not os.path.exists('./cola_public/'):
#    !unzip cola_public_1.1.zip



#2. Reading data
import pandas as pd

dataframe = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
print('Number of training sentences: {:,}\n'.format(dataframe.shape[0]))

dataframe.sample(10)
#Display 10 random rows of data (only shows in notebook)

sentences = dataframe.sentence.values
labels = dataframe.label.values



#3. Tokenize and input formatting
from transformers import BertTokenizer

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


#Visualizing tokenizer
"""
# Print the original sentence.
print(' Original: ', sentences[0])
# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))
# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
"""

#BERT specific tokens
#CLS to mark beginning of sentence pair, and contiains hidden states of sentence during training, SEP marks end of sentence

#Sentence length and attention mask
#sentences have a max lenght and those with less words will have empty "pad" tokens 
#mask is redundent array indicating if token is real word, or padding

#finding max sentence length 
maxLength = 0

for sentence in sentences:
    inputID = tokenizer.encode(sentence, add_special_tokens=True)
    maxLength = max(maxLength, len(inputID))

print(maxLength)


#tokenize all sentences and map tokens to word ID
inputID = []
attentionMasks = []

for sentence in sentences:
    encodedDict = tokenizer.encode_plus(
        sentence,
        add_special_tokens = True,
        max_length = 64,
        padding='max_length',
        return_attention_mask = True,
        return_tensors = 'pt',
        truncation=True
    )

    #Adding encoded sentence to the list
    inputID.append(encodedDict['input_ids'])

    #add attnetion mask to list
    attentionMasks.append(encodedDict['attention_mask'])

#Convert lists into tensors
inputID = torch.cat(inputID, dim=0)
attentionMasks = torch.cat(attentionMasks, dim=0)
labels = torch.tensor(labels)

print('Original: ', sentences[0])
print('Token IDs: ', inputID[0])
