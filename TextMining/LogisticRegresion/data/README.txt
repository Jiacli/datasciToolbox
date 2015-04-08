Data description

This dataset is a set of paper abstracts from the CITESEER dataset. It has 17 class-labels which refer to one more the research topics such as AI, Machine Learning, Architecture, etc. There are 2865 training documents and 1432 testing documents with a vocabulary size of 14,601 words. For this assignment, you do not need to worry what each word is or what each topic represents (just assume that there are 17 topics 1 to 17 and 14,601 words from 1 to 14,601). 

Format

The training and testing file share the same format. Each line represents one abstract. The first integer in the line represents the class-label of the abstract. Next follows the contents of the abstract in a bag of words representation. The value of each word is the normalized tf-idf value. Words which does not show in the bag of words representation have a value of zero. 
For example, the line

15 3:.23 435:1.23 1337:3.66 

says that the article belongs to topic 15. It has three words 3, 435 and 1337 whose values are .23, .23 and 3.66 repspectively. All the words do not occur and have the value of zero.

Other Notes

1. All the words in the representation are in increasing order.
2. Each abstract belongs to exactly one topic.

