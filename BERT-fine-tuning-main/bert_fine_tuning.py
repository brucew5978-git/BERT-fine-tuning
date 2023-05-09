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

#print('Original: ', sentences[0])
#print('Token IDs: ', inputID[0])


#Dividing data into 90% training, 10% validation
from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(inputID, attentionMasks, labels)

trainSize = int(0.9 * len(dataset))
valSize = len(dataset) - trainSize

trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

print('{:>5,} training samples'.format(trainSize))
print('{:>5,} validation samples'.format(valSize))


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batchSize = 32

#Creates training samples in random order
trainDataloader = DataLoader(
    trainDataset,
    sampler = RandomSampler(trainDataset),
    batch_size=batchSize
)
#Order will not matter for validation
validationDataloader = DataLoader(
    valDataset,
    sampler=SequentialSampler(valDataset),
    batch_size=batchSize
)



#4. Training classification model
from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2, #number of output labels
    output_attentions=False,
    output_hidden_states=False
)

#model.cuda()

'''
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())
print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
'''


#Optimization and training params
'''
will initially use following training vars
Batch size: 32 (set when creating our DataLoaders)
Learning rate: 2e-5
Epochs: 4 (we'll see that this is probably too many...)
'''

optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps = 1e-8
                )

from transformers import get_linear_schedule_with_warmup
trainingEpochs = 4

totalSteps = len(trainDataloader)*trainingEpochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=totalSteps)


#Training
import numpy as np

def flatAccuracy(preds, labels):
    predFlat = np.argmax(preds, axis=1).flatten()
    labelsFlat = labels.flatten()
    return np.sum(predFlat == labelsFlat) / len(labelsFlat)

import time
import datetime

def formatTime(elapsed):
    elapsedRounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsedRounded))

import random 
import numpy as np

seedVal = 42

random.seed(seedVal)
np.random.seed(seedVal)
torch.manual_seed(seedVal)
torch.cuda.manual_seed_all(seedVal)

trainingStats = []

totalTime = time.time()

for epoch in range(0, trainingEpochs):
    print('/n ===== Epoch {:} / {:} ====='.format(epoch + 1, trainingEpochs))
    print("training")

    currentEpochTime = time.time()

    totalTrainLoss = 0

    model.train()

    for step, batch in enumerate(trainDataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = formatTime(time.time() - currentEpochTime)

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(trainDataloader), elapsed))

        #copy tensor to GPU using 'to' method
        bathcInputID = batch[0].to(device)
        batchInputMask = batch[1].to(device)
        batchLabels = batch[2].to(device)

        #clearing previous gradients before backward pass
        model.zero_grad()

        results = model()


