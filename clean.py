import re
import tensorflow as tf
import pandas as pd
import numpy as np
from data_processing import padding_sequences



class cleaner(object):

	def __init__(self, uquestion, word_index_path, max_length=58):

		self.question = uquestion
		self.cleaned = self.real_clean(self.question).split()
		self.word_index, self.index_word = self.vocabIndexing(word_index_path)


		# print self.cleaned
		# print type(self.cleaned)

		self.cleaned_unpadded_question = self.question2indices(self.cleaned)

		self.as_tagger = ' '.join(map(self.index_word.get, self.cleaned_unpadded_question))

		# print self.cleaned_unpadded_question


		self.length = len(self.cleaned_unpadded_question)



		self.question_as_placeholder = padding_sequences([self.cleaned_unpadded_question], max_length)


	def as_inputs(self):
		return self.question_as_placeholder.tolist(), np.array([self.length])

	def real_clean(self, unclean):
		unclean = self.repairsymbol(unclean, '?')
		unclean = unclean.replace('"','')
		unclean = re.sub(r"\(s\)", r"s", unclean)
		unclean = self.repairsymbol(unclean, '<')
		unclean= self.repairsymbol(unclean, '>')
		unclean= self.repairsymbol(unclean, '/')
		unclean= self.repairsymbol(unclean, ',')
		unclean= self.repairsymbol(unclean, ':')
		unclean= self.repairsymbol(unclean, ']')
		unclean= self.repairsymbol(unclean, '[')
		unclean= self.repairsymbol(unclean, '}')
		unclean= self.repairsymbol(unclean, '{')
		unclean= self.repairsymbol(unclean, '-')
		unclean= self.repairsymbol(unclean, '&')
		unclean= self.repairsymbol(unclean, '$')
		unclean=unclean.lower()
		unclean= re.sub(r"what's", r"what is", unclean)
		unclean=re.sub(r"(python) (3\..)", r"\1\2", unclean)
		unclean=re.sub(r"(python) (2\..)", r"\1\2", unclean)
		unclean= re.sub(r"(\*{1,})([a-zA-Z]{2,})(\1)", r"\2", unclean)
		unclean= self.repairsymbol(unclean,'\\')
		unclean= re.sub(r"(\()([a-zA-Z \.]+)(\))", r" \1 \2 \3 ", unclean)
		unclean= self.repairsymbol(unclean, ';')
		unclean=self.repairsymbol(unclean, '@')
		unclean= self.repairsymbol(unclean, '=')
		unclean= re.sub(r"[ ]{1,}", r" ", unclean)      

		return unclean # but is cleaned now !!                               


	def repairsymbol(self, x, symbol):
		'''convert "????" to "? ? ? ? ?" '''
		return re.sub(r"[ ]{1,}", r" ", (' '+symbol+' ').join(x.split(symbol)))



	def vocabIndexing(self, dir_path):
	    vocabfile = dir_path+'vocab.txt' #'vocab.txt'
	    vocab = []

	    with open(vocabfile, 'r') as f:
	        for line in f:
	                vocab.append(line.split()[0])

	    vocab2index = {v:i for i,v in enumerate(vocab) }
	    index2vocab = {i:v for i,v in enumerate(vocab) }

	    return vocab2index, index2vocab




	def question2indices(self, question):
	        """
	        takes a question and converts it to indices of words in it.
	        words not found in word_index are simply ignored at this level.

	        """
	        return [self.word_index[word] for word in question if self.word_index.get(word) is not None]



if __name__ == '__main__':

	cq = cleaner("What is function of apply in pandas?", '')
	# print cq.as_inputs()[1].shape
	print cq.as_tagger