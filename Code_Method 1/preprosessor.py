import logging
import pandas as pd
import re
from nltk.corpus import stopwords
import pickle
from pathlib import Path


logger = logging.getLogger(__name__)

def clean_text(raw_text):

	# keep only words
	letters_only= re.sub("[^a-zA-Z]", " ", raw_text)

	# convert to lower case and split 
	words = letters_only.lower().split()

	cleaned_word_list = " ".join(words)

	return raw_text


#读取数据（review & label），使用clean_text处理并记录数据处理数量
def preprocess(dataset, is_train = True):
	
	logger.debug("Beginning processing of reviews")

	reviews_df = pd.read_csv(dataset,delimiter='\t')
	num_reviews = reviews_df.shape[0]
	cleaned_reviews = []
	cleaned_reviews_labels = []

	logger.info("Total reviews before beginning the cleaning process: " + str(num_reviews))
	
	empty_reviews_count = 0
	cleaned_reviews_count = 0

	for i in range(num_reviews):
		review = reviews_df.iloc[i][2]	# 索引数据，通过行号获取行数据，不能是字符
		old_len = len(review)
		cleaned_review = clean_text(review)
		
		if len(cleaned_review) == 0:
			# if the cleaned review comes empty then add a dummy token
			cleaned_review = "<DUMMYDUMMY>"	
			empty_reviews_count += 1
		else:
			cleaned_reviews_count +=1

		cleaned_reviews.append(cleaned_review)
		
		if is_train == True:	#如果被告知是训练集，则读取并存入标签
			cleaned_reviews_labels.append(reviews_df.iloc[i][3])	#标签位置的数据

		if(i % 20000 == 0):
			logger.info(str(i) + " reviews processed")
		
	logger.debug("Total zero length reviews after cleaning " + str(empty_reviews_count))
	logger.debug("Total reviews after cleaning process is finished: " + str(cleaned_reviews_count))
	logger.debug("Finished processing of reviews")

	if is_train == True :
		return cleaned_reviews, cleaned_reviews_labels
	else: 
		return cleaned_reviews

#pickle
def preprocess_dataset(dataset_file, is_train):
	
	logger.info("Begin preprocessing of :" + str(dataset_file))

	cleaned_reviews, cleaned_reviews_labels = None, None

	if is_train == True :
		words_pickle = "cleaned_train_dataset.pkl"
		labels_pickle = "cleaned_train_labels.pkl"
	else:
		words_pickle = "cleaned_test_dataset.pkl"

	if Path(words_pickle).is_file() == False:
		# Option 1: Load CSV, clean it and dump it in a pickle
		logger.info("No dataset pickle file found. Generating it now.")
		logger.info("Loading dataset from csv and cleaning it : " + dataset_file)
		
		if is_train == True:
			cleaned_reviews, cleaned_reviews_labels = preprocess(dataset_file, is_train)
			labels_pickle = open(labels_pickle,'wb')
			pickle.dump(cleaned_reviews_labels, labels_pickle)     #将obj对象序列化存入已经打开的file中
		else:
			cleaned_reviews = preprocess(dataset_file, is_train)
		
		words_pickle = open(words_pickle,'wb')
		pickle.dump(cleaned_reviews, words_pickle)

	else:
		# Option 2: Load an existing pickle，继续在之前的pickle上工作
		logger.info("Dataset and label pickle files found. Loading them now.")
		logger.info("Loading cleaned dataset from pickle file : " + words_pickle)
		cleaned_reviews = pickle.load(open(words_pickle,'rb'))

		
		if is_train == True:
			logger.info("Loading corresponding labels from pickle file : " + labels_pickle)
			x = pickle.load(open(labels_pickle,'rb'))
	
	logger.info("Finished preprocessing of : " + str(dataset_file))
	
	if is_train == True :
		return cleaned_reviews, cleaned_reviews_labels
	else: 
		return cleaned_reviews