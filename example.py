from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import os
            
""" ham_path = "./data/data/myham/"
vectorizer = CountVectorizer()

file_list = os.listdir(ham_path)
content = []

for file in file_list:
    with open(ham_path + file) as f:
        content.append(f.read())
        
words = vectorizer.fit_transform(content)
#print(vectorizer.vocabulary_)

namedict = dict()
words_array = words.toarray()
likehood = np.zeros(len(words_array[0]))

for word in vectorizer.vocabulary_:
    namedict[vectorizer.vocabulary_[word]] = word
    
for fileindex in range(len(words_array)):
    for index in range(len(words_array[0])):
        likehood[index] = likehood[index] + words_array[fileindex][index]

for index in range(len(likehood)):
    print(namedict[index],likehood[index]) """
    
ham_dict = {'able': 1.0, 'about': 3.0, 'after': 1.0, 'alice': 1.0, 'am': 3.0, 'and': 3.0, 'answer': 1.0, 'as': 2.0, 'assignment': 1.0, 'bar': 1.0, 'be': 1.0, 'best': 4.0, 'can': 1.0, 'class': 4.0, 'could': 1.0, 'course': 2.0, 'dear': 3.0, 'do': 1.0, 'earlier': 1.0, 'email': 1.0, 'exam': 3.0, 'find': 1.0, 'fits': 1.0, 'follow': 1.0, 'foo': 1.0, 'forward': 3.0, 'from': 3.0, 'further': 1.0, 'getting': 1.0, 'hanks': 1.0, 'have': 2.0, 'hearing': 2.0, 'in': 4.0, 'incorporate': 1.0, 'is': 1.0, 'it': 1.0, 'kim': 1.0, 'kumar': 1.0, 'last': 1.0, 'later': 1.0, 'learning': 1.0, 'like': 1.0, 'listed': 1.0, 'long': 1.0, 'look': 3.0, 'machine': 1.0, 'meet': 1.0, 'michael': 1.0, 'more': 1.0, 'my': 3.0, 'not': 1.0, 'of': 4.0, 'on': 3.0, 'one': 1.0, 'only': 1.0, 'or': 1.0, 'our': 1.0, 'part': 1.0, 'professor': 3.0, 'project': 2.0, 'question': 2.0, 'realized': 1.0, 'regarding': 1.0, 'regards': 3.0, 'reply': 1.0, 'results': 1.0, 'review': 1.0, 'second': 2.0, 'sent': 1.0, 'should': 1.0, 'soo': 1.0, 'struggling': 1.0, 'student': 2.0, 'subject': 1.0, 'syllabus': 2.0, 'talk': 1.0, 'term': 1.0, 'than': 1.0, 'that': 1.0, 'the': 13.0, 'there': 1.0, 'this': 3.0, 'time': 1.0, 'to': 8.0, 'topic': 2.0, 'topics': 2.0, 'up': 1.0, 'want': 1.0, 'was': 1.0, 'we': 1.0, 'week': 3.0, 'whole': 1.0, 'with': 3.0, 'would': 1.0, 'writing': 1.0, 'you': 3.0, 'your': 3.0}
spam_dict = {'000': 2.0, '1999': 1.0, '20': 1.0, '31': 1.0, 'ab': 1.0, 'about': 1.0, 'advertisement': 1.0, 'after': 1.0, 'already': 1.0, 'an': 2.0, 'analysis': 3.0, 'and': 3.0, 'announce': 1.0, 'anything': 1.0, 'aol': 1.0, 'are': 2.0, 'as': 2.0, 'attempts': 1.0, 'available': 1.0, 'be': 3.0, 'been': 1.0, 'build': 1.0, 'business': 1.0, 'can': 4.0, 'completed': 1.0, 'contest': 1.0, 'could': 1.0, 'day': 1.0, 'days': 1.0, 'decided': 1.0, 'definitely': 1.0, 'dollars': 1.0, 'each': 1.0, 'email': 1.0, 'equity': 1.0, 'even': 1.0, 'exactly': 1.0, 'fastest': 1.0, 'for': 4.0, 'forwards': 1.0, 'free': 2.0, 'friends': 2.0, 'give': 2.0, 'good': 1.0, 'have': 2.0, 'how': 3.0, 'idea': 1.0, 'in': 2.0, 'inc': 1.0, 'internet': 2.0, 'is': 3.0, 'it': 4.0, 'january': 1.0, 'just': 1.0, 'lose': 1.0, 'luck': 1.0, 'lucky': 1.0, 'mail': 1.0, 'mailed': 1.0, 'mailing': 1.0, 'make': 1.0, 'many': 3.0, 'me': 1.0, 'money': 1.0, 'mortgage': 2.0, 'much': 1.0, 'new': 1.0, 'no': 2.0, 'obligation': 1.0, 'of': 4.0, 'off': 1.0, 'on': 1.0, 'once': 1.0, 'one': 2.0, 'or': 1.0, 'our': 1.0, 'people': 1.0, 'period': 1.0, 'picked': 1.0, 'profitable': 1.0, 'program': 1.0, 'proud': 1.0, 'questions': 1.0, 'randomly': 1.0, 'receive': 1.0, 'return': 1.0, 'run': 1.0, 'says': 1.0, 'send': 1.0, 'sending': 1.0, 'start': 1.0, 'take': 1.0, 'talking': 1.0, 'tell': 1.0, 'thank': 1.0, 'the': 5.0, 'their': 1.0, 'these': 1.0, 'thing': 1.0, 'this': 2.0, 'though': 1.0, 'thousands': 1.0, 'through': 1.0, 'time': 3.0, 'to': 8.0, 'trust': 1.0, 'try': 1.0, 'trying': 1.0, 'under': 1.0, 'until': 1.0, 'user': 1.0, 'we': 3.0, 'weeks': 1.0, 'what': 1.0, 'who': 1.0, 'will': 5.0, 'winner': 1.0, 'with': 1.0, 'within': 1.0, 'you': 12.0, 'your': 4.0, 'yrs': 1.0}

sorted_ham = sorted(ham_dict.items(),key = lambda item: item[1],reverse = True)
sorted_spam = sorted(spam_dict.items(),key = lambda item: item[1],reverse = True)
print(sorted_ham)
print(sorted_spam)