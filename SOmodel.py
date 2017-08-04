"""Usage:
  SOmodel.py train [options]
  SOmodel.py [options]
  SOmodel.py -h | --help | --version

  options:
    -r, --reuse       for reuse 
    -e, --eval        for eval
    -l, --live        for live
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pylab import plt
from docopt import docopt
import json

from data_processing import load_embeddings, load_and_preprocess_data,tag2label, label2tag, labels_list, Ldict

from clean import cleaner
from hyperparameters import Config

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.preprocessing.sequence import pad_sequences


class Tagger(object):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will produce a tag for a question from stackoverflow.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length,), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        TODO: Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        HINTS:
            - Remember to use self.max_length NOT Config.max_length

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~4-6 lines)
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.max_length),name='input_placeholder')
        self.seqlen_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,), name='seqlen_placeholder')
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,),name='labels_placeholder')
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32,name='dropout_placeholder')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.L = tf.placeholder(dtype=tf.float32, shape=(None, ), name='L')

        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, seqlen_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)
        feed_dict = {self.input_placeholder:inputs_batch, self.dropout_placeholder:dropout, 
                                self.seqlen_placeholder: seqlen_batch}

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch 
            labels_batch = labels_batch.reshape(-1)            # (4,1) to (4,) and (4,) to (4,)
            feed_dict[self.L] = map(lambda x: 0.1/Ldict.get(label2tag.get(x)), labels_batch)

        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors.

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        embedding = tf.get_variable("embedding",
                                    initializer=self.pretrained_embeddings)

        # print self.input_placeholder
        embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, self.input_placeholder) 
        # rows have the embedding vectors.

        ### END YOUR CODE
        # print embeddings.get_shape().as_list()
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        TODO: There a quite a few things you'll need to do in this function:
            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
              https://www.tensorflow.org/api_docs/python/constant_op/constant_value_tensors#zeros
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#shape
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
              See https://www.tensorflow.org/versions/master/how_tos/variable_scope/
            - Concatenate and reshape the predictions into a predictions
              tensor.
        Hint: You will find the function tf.pack (similar to np.asarray)
              useful to assemble a list of tensors into a larger tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#pack
        Hint: You will find the function tf.transpose and the perms
              argument useful to shuffle the indices of the tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#transpose

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        preds = [] # Predicted output at each timestep should go here!

        # Use the cell defined below. For Q2, we will just be using the
        # RNNCell you defined, but for Q3, we will run this code again
        # with a GRU cell!
        if self.config.cell == "rnn":
            cell = RNNCell(Config.n_features * Config.embed_size, Config.hidden_size)
        elif self.config.cell == "gru":
            cell = GRUCell(Config.n_features * Config.embed_size, Config.hidden_size)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        ### YOUR CODE HERE (~4-6 lines)
        U = tf.get_variable(name='U',shape=(self.config.hidden_size,self.config.n_classes),
            initializer=tf.contrib.layers.xavier_initializer() )
        b_2 = tf.get_variable(name='b_2', shape=(self.config.n_classes),
            initializer=tf.contrib.layers.xavier_initializer())
        # state = tf.constant(0.0, shape=[None, self.config.hidden_size], dtype=tf.float32)
        state = tf.zeros(shape=[tf.shape(self.input_placeholder)[0], self.config.hidden_size])
        ### END YOUR CODE

        with tf.variable_scope("RNN"):
            for time_step in range(self.max_length):
                ### YOUR CODE HERE (~6-10 lines)

                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                out, state = cell(x[:,time_step,:], state, scope=tf.get_variable_scope())
                outDrop = tf.nn.dropout(out, dropout_rate)
                preds.append(tf.matmul(outDrop, U) + b_2)
                ### END YOUR CODE

        # Make sure to reshape @preds here.
        ### YOUR CODE HERE (~2-4 lines)
        preds = tf.stack(preds, axis=1)
        # preds = tf.reshape(preds, (None, self.max_length, self.config.n_classes))
        ### END YOUR CODE

        assert preds.get_shape().as_list() == [None, self.max_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds


    def add_prediction_op_from_tensorflow(self):
        x = tf.cast(self.add_embedding(), tf.float32)
        # print type(x)
        # assert False

        print x.dtype

        cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, sequence_length=self.seqlen_placeholder,dtype=tf.float32)
        # rnn_outputs should have shape of (None, self.config.max_length, self.config.hidden_size)
        rnn_outputs = tf.nn.dropout(rnn_outputs, self.dropout_placeholder)
        average_rnn_outputs = tf.reduce_mean(rnn_outputs, axis=1)

        W1 = tf.get_variable(name='W1',shape=(self.config.hidden_size, self.config.n_classes),
            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b1 = tf.get_variable(name='b1', shape=(self.config.n_classes),
            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

        preds = tf.matmul(average_rnn_outputs, W1)+b1

        # assert preds.get_shape() == (?, self.config.n_classes)
        return preds
        



    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        TODO: Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Hint: You might find tf.boolean_mask useful to mask the losses on masked tokens.
        Hint: You can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-4 lines)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=self.labels_placeholder)
        # loss = tf.reduce_mean(loss)

        loss = tf.losses.sparse_softmax_cross_entropy(logits=preds, labels=self.labels_placeholder,
                                                    weights = self.L,
                                                    reduction=tf.losses.Reduction.MEAN)

        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss, global_step=self.global_step)
        ### END YOUR CODE
        return train_op

    def predict_on_batch(self, sess, inputs_batch, seqlen_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, seqlen_batch=seqlen_batch)
        predictions_proba = sess.run(self.pred, feed)

        # predictions, pred_proba = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return np.argmax(predictions_proba, axis=1), predictions_proba

    def train_on_batch(self, sess, inputs_batch, labels_batch, seqlen_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, seqlen_batch=seqlen_batch,
                                     dropout=Config.dropout)
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed)
        return loss, summary


    def create_summaries(self, loss):
        with tf.variable_scope("summaries"):
            tf.summary.scalar("loss",loss)
            tf.summary.histogram("histogram loss", loss)
            summary_combine = tf.summary.merge_all()

        return summary_combine

    def __init__(self, config, pretrained_embeddings):
        self.config = config
        self.max_length = Config.max_length
        # Config.max_length = self.max_length # Just in case people make a mistake.

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.seqlen_placeholder = None
        self.dropout_placeholder = None

        self.add_placeholders()
        self.pretrained_embeddings = pretrained_embeddings

        self.pred = self.add_prediction_op_from_tensorflow()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.summary_op = self.create_summaries(self.loss) # this also an op


def generate_minibatches(examples, batch_size, shuffle=True):
    """
    examples: [(question, length, label), ...]
             question: list of indices of words appearing in question.

    for q, len, label in generate_minibatches(examples[train/dev]):
        .....
        q, len, label have first dimension as None or batch_size.

    NOTE: use yield here.
          np.array().tolist() gives python list.

    shuffle: True (default), should be used when training is happening.

    returns q, len, label where all of these are numpy arrays.
    """
    if type(examples) is list:
        examples = np.array(examples)

    num_examples = examples.shape[0] ; indices = np.arange(num_examples) ; np.random.shuffle(indices)
    examples = examples[indices]
    
    for i in range(0, num_examples, batch_size):
        yield map(np.array, zip(*examples[i: i+batch_size]))


def classification_metrics(y_true,y_pred,label2tag=label2tag,
    labels_list=labels_list,path=Config.output_path,gstep=1000):
    """
    y_target : list of numpy array or list with integers representing classes.
    y_pred   : samea as y_target
    labels_list: in order to print the scores. 
    label2tag : is dict like {0:others, ...}

    prints 
    """
    y_true_tag = map(label2tag.get, y_true)
    y_pred_tag = map(label2tag.get, y_pred)

    f1 = f1_score(y_true_tag, y_pred_tag, labels=labels_list, average=None)
    pm = precision_score(y_true_tag, y_pred_tag, labels=labels_list, average=None)
    rm = recall_score(y_true_tag, y_pred_tag, labels=labels_list, average=None)

    # Just for pretty printing using pandas 
    metric_df = pd.DataFrame({'f1':f1, 'pm':pm, 'rm': rm}, index=labels_list)
    print metric_df
    metric_df.to_csv(path+"f1score_"+str(gstep)+".csv")

    # HERE WE HAVE SERIOUS BUG. ONLY WORKS WHEN ALL OF THEM ARE PREDICTED PROPERLY.
    # Could be possible :if there is no some tag's data in dev set.
    # checking dataproportion error here..   but its fixed now I jesss 


    y_true_set = set(y_true) ; y_pred_set = set(y_pred)
    # print y_true_set
    # print y_pred_set

    data_proportion_error = y_true_set - y_pred_set
    
    if len(data_proportion_error) > 0:
        print "Some classes are not predicted by classifier."
        # set(y_true)
    
    labels_list = map(label2tag.get, sorted(list(y_true_set)) )
    # print labels_list

    confu = confusion_matrix(y_true, y_pred)
    # print confu
    
    # plotting business    
    fig = plt.figure(gstep, figsize=(20,20))    
    sns_plot = sns.heatmap(confu, annot=True, xticklabels=labels_list, yticklabels=labels_list, fmt=".0f")
    confu_path = path+"confusion_matrix"+str(gstep)+".png" 
    # print path, confu_path

    plt.savefig(confu_path)


def do_train():
    # Set up some parameters.
    config = Config()    
    train, dev, max_length = load_and_preprocess_data(Config.dir_path,should_split=True,prop=0.8)
    Config.max_length = 58 #max_length
    config.store_config_info()

    num_train_examples = len(train[0])
    num_dev_examples = len(dev[0])
    print "loaded & processed", len(train[0]), "train examples"
    print "loaded & processed", len(dev[0]), "dev examples"
    print "max_length for complete data is:", Config.max_length

    embeddings = load_embeddings(Config.dir_path)

    with tf.Graph().as_default():
        model = Tagger(config, embeddings)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep = 10)

        with tf.Session() as session:
            session.run(init)

            # This is used to restore the model but the new params will be stored in the current one.
            # I am not implementing that, for now because I have my reasons.

            writer = tf.summary.FileWriter(Config.output_path, session.graph)


            if Config.previous_path:
                print "using previous_path aka params are restored."
                checkPoint = tf.train.get_checkpoint_state(Config.previous_path)

                if checkPoint and checkPoint.model_checkpoint_path:
                    saver.restore(session, checkPoint.model_checkpoint_path)

                outpath = Config.previous_path
                checkname = 'checkpoint1'
            else:
                print "Not using previous_path aka training from start"
                outpath = Config.output_path
                checkname = 'checkpoint'
                config.create_output_folder()

            print outpath

            for epoch in xrange(model.config.n_epochs):
                for i, batched in enumerate(generate_minibatches(train[0], model.config.batch_size)):
                    q, lngth, label = batched
                    loss_per_e, summary_per_e = model.train_on_batch(session, q, label, lngth)
                    writer.add_summary(summary_per_e, global_step=model.global_step.eval())
                    print epoch+1, loss_per_e


                saver.save(session, outpath, global_step=model.global_step, latest_filename=checkname)
                print "Session is saved"

                # here evaluating.
                continue
                y_pred_tag = [] ; y_true_tag = []
                for i,batched in enumerate(generate_minibatches(dev[0], 128, False)):

                    q, lngth, label = batched
                    predict_proba, predict_class = model.predict_on_batch(session, q, lngth).tolist()
                    y_pred_tag += predict_class
                    y_true_tag += label.tolist()

                    # if i==5:
                    #     break                

                # assert len(y_pred_tag) == len(y_true_tag)
                classification_metrics(y_true_tag, y_pred_tag, path=outpath, gstep=model.global_step.eval())

                # This is for my laptop's 
                # if epoch == 4:
                #     break    


def do_evaluate():
    """
    For now it just evaluates on my dev-set
    It is for a general dev set.
    """

    config = Config(False)    
    _, dev, max_length = load_and_preprocess_data(Config.dir_path,should_split=True,prop=0.8)
    Config.max_length = max_length
    num_dev_examples = len(dev[0])

    print "loaded & processed", num_dev_examples, "dev examples"
    print "max_length for complete data is:", max_length

    embeddings = load_embeddings(Config.dir_path)

    with tf.Graph().as_default():
        model = Tagger(config, embeddings)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)

            checkPoint = tf.train.get_checkpoint_state(Config.eval_path, latest_filename='checkpoint')

            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(session, checkPoint.model_checkpoint_path)

                # here evaluating.
                # print "here"s

                y_pred_tag = [] ; y_true_tag = []
                for i,batched in enumerate(generate_minibatches(dev[0], 128, False)):
                    s1 = time.time()
                    q, lngth, label = batched
                    predict_class, predict_proba = model.predict_on_batch(session, q, lngth)
                    y_pred_tag += predict_class.tolist()
                    y_true_tag += label.tolist()

                    # print "batch {} out of {} finished in {} with max_len:{}".\
                    #             format(i, num_dev_examples/(1.0*model.config.batch_size), (time.time()-s1), max(lngth))
 
                # assert len(y_pred_tag) == len(y_true_tag)
                classification_metrics(y_true_tag, y_pred_tag, path=Config.eval_path, gstep=model.global_step.eval())




def do_live():
    config = Config(False)
    embeddings = load_embeddings(Config.dir_path)

    with tf.Graph().as_default():
        model = Tagger(config, embeddings)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)

            # This is used to restore the model but the new params will be stored in the current one.
            # I am not implementing that, for now because I have my reasons.
            checkPoint = tf.train.get_checkpoint_state(Config.eval_path)

            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(session, checkPoint.model_checkpoint_path)

            print "===================================================================================="
            print "===================================================================================="

            print "> Enter the question..  to exit type EXIT"
            print ""
            print "> ",

            # question_input = raw_input().lower()
            question_input = "What is dataFrame in pandas ?"
            while True:

                if question_input == 'exit':
                    break

                question_words_clean, seqlen = cleaner(question_input, Config.dir_path).as_inputs() 
                # assert question_words_clean.shape[0] == 1
                # assert seqlen.shape[0] == 1

                cleanerPro = cleaner(question_input, Config.dir_path)
                question_words_clean, seqlen = cleanerPro.as_inputs()


                # print type(question_words_clean), type(seqlen)
                print " "
                print "========================================================="
                print "> your question interpreted as :"
                print ">",cleanerPro.as_tagger
                print "> predicted tag is: ",
                predict_class, predict_proba = model.predict_on_batch(session, question_words_clean, seqlen)

                # print predict_class

                print label2tag.get(predict_class[0])
                print ""
                max_prob_indx = np.argsort(-1.0*np.array(predict_proba))[0][:5]

                print "per class probabilities"
                print map(label2tag.get, max_prob_indx)
                # print predict_proba
                print np.array(np.exp(predict_proba[0])/np.sum(np.exp(predict_proba[0])))[max_prob_indx]
                

                print "========================================================="
                print " "

                print "> ",
                question_input = raw_input().lower()

#What does "use strict" do in JavaScipt, and what is the reasoning behind it?


if __name__ == '__main__':

    arguments = docopt(__doc__)

    if arguments['--eval']:
        do_evaluate()
    elif arguments['--live']:
        do_live()
    elif arguments['train']:
        Config.previous_path = None if not arguments["--reuse"] else Config.previous_path
        do_train()
    else:
        print "type this: SOmodel1.py -h"
