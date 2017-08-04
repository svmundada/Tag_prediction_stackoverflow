import numpy as np
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from hyperparameters import Config

labels_list = ['others',
 'python',
 'php',
 'sql',
 'c',
 'html',
 'c#',
 'java',
 'javascript',
 'c++',
 'css',
 'regex',
 'android',
 'jquery',
 'arrays',
 'ios']


Ldict = {'android': 0.16216627,
 'arrays': 0.013860364,
 'c': 0.13291787,
 'c#': 0.32543528,
 'c++': 0.26039276,
 'css': 0.048511274,
 'html': 0.027629675,
 'ios': 0.14167923,
 'java': 0.51478606,
 'javascript': 0.2635189,
 'jquery': 0.041570976,
 'others': 1.0,
 'php': 0.26137409,
 'python': 0.33866841,
 'regex': 0.026253756,
 'sql': 0.10257681}

label2tag = {i:v for i, v in enumerate(labels_list)} # this should maintain sorted for confusion_matrix thing.
tag2label = {v:i for i, v in enumerate(labels_list)}


def load_embeddings(dir_path):
    """ returns embeddings matrix,
        embeddings_matrix.shape[0] = Total vocabulary.
        embeddings_matrix.shape[1] = embed_size. 


        This method will be called for pretrained_embeddings only.

    """
    
    def vocabIndexing(VOCAB_PATH):
        vocabfile = VOCAB_PATH #'vocab.txt'
        vocab = []

        with open(vocabfile, 'r') as f:
            for line in f:
                    vocab.append(line.split()[0])

        vocab2index = {v:i for i,v in enumerate(vocab) }
        index2vocab = {i:v for i,v in enumerate(vocab)}

        return vocab2index, index2vocab


    def word_vector_mapping(VECTORS_PATH):
        """ maps words to vectors and returns: a dict
        like {word1:vector1}
        """
        wordvfile = VECTORS_PATH

        word2vector = {}

        with open(wordvfile, 'r') as f:
            for line in f:
                combo = line.split()
                word, vector = combo[0], map(float,combo[1:])
                word2vector[word] = np.array(vector)

        return word2vector
    

    def make_embedding_matrix(word_index, embeddings_index, EMBEDDING_DIM=50):    
        """ returns embedding matrix to be used in keras.
            mapping: word_index --> word_vector
        """
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

        for word, i in word_index.items():
                embedding_matrix[i] = embeddings_index[word]
                
        return embedding_matrix




    word_index = vocabIndexing(dir_path+'vocab.txt')[0]

    embeddings_index = word_vector_mapping(dir_path + 'vectors.txt')

    return make_embedding_matrix(word_index, embeddings_index)


def load_and_preprocess_data(dir_path, should_split=True, prop=0.8):
    """
    return train, dev examples.
        first training examples(both processed and raw in this order) and then dev (as previous tuple.)
        
        updated for top6: returns max_length also.
    
    train = [(question, length, label), ]
    same dev also.

    question: is the padded index of words appearing in the raw_question.
    length: Length of padded questions.
    label: one the twenty labels. It is a scalar.

    NOTE: Call padding_sequences method.


    personal: also should have an inner function to change questions to word_indices
            : Tricky: words not appearing in word vocabulary should be directly ignored.
              consequ: This could drastically shorten the length of questions and length will also  
                        change here. So this function is to applied at very first while 
                        reading data or 
                

            : and then apply padding sequences on them.

    """
    def vocabIndexing(dir_path):
        vocabfile = dir_path+'vocab.txt' #'vocab.txt'
        vocab = []

        with open(vocabfile, 'r') as f:
            for line in f:
                    vocab.append(line.split()[0])

        vocab2index = {v:i for i,v in enumerate(vocab) }

        return vocab2index

    word_index = vocabIndexing(dir_path)
        
    def question2indices(question):
        """
        takes a question and converts it to indices of words in it.
        words not found in word_index are simply ignored at this level.

        """
        return [word_index[word] for word in question if word_index.get(word) is not None]
        

    df = pd.read_csv(dir_path+'data_top16.csv', 
                     converters={'Title_list':literal_eval}, engine='c', usecols=['Title_list', 'label'])
    
    questions =  df['Title_list']
    label = df['label']

    questions_indices = map(question2indices, questions)
    length  = map(len, questions_indices)

    # here i forcing fixed max_length of my wish.
    Config.max_length = 58
    # now padding  since original length will be fed to seqlen.
    questions_indices_padded = padding_sequences(questions_indices, Config.max_length)





    # here data should be processed very well.
    # planning to return raw also
    data = np.array(zip(*[questions_indices_padded, length, label]))
    data_raw = np.array(zip(*[questions, length, label]))
    num_data = len(data)
    indices = np.arange(num_data) ; np.random.shuffle(indices)
    data = data[indices] ; data_raw = data_raw[indices]

    if should_split:
        num_train = int(num_data*prop)
        return ((data[:num_train], data_raw[:num_train]), (data[num_train:], data_raw[num_train:]), Config.max_length)
    else:
        return data, data_raw, Config.max_length


def padding_sequences(sequences, max_length):
    """
    Truncates and Pads to max_length.
    returns: padded sequences.

    NOTE: try to use keras's padding sequence.
           

    """

    # padding is pre and truncation is also pre.
    return pad_sequences(sequences, max_length,padding='post', truncating='post')



