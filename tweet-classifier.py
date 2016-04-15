import nltk
import re
import random

# random.seed(335569132214972416) # id of kanye's first tweet
random.seed(654455694880567296) # id of trump's first tweet

# read in text
kanye_tweets = open('kanyewest_tweets.csv').read().splitlines()
trump_tweets = open('realDonaldTrump_tweets.csv').read().splitlines()

# label data
documents = ( [(row, 'kanye') for row in kanye_tweets ] +
              [(row, 'trump') for row in trump_tweets ])

def is_stop_word(word):
  return word.startswith('http') or word.startswith('#') or word.startswith('@') or ('kanye' in word) or ('trump' in word) or re.match('\d', word)

# lowercase words, exclude stop words
valid_words = set()
for (tweet, label) in documents:
  for word in tweet.split():
    word = re.sub('\W', '', word).lower()
    if not is_stop_word(word):
      valid_words.add(word)

def document_features(document):
  document_words = document.split()
  features = {}
  for word in valid_words:
    features['contains({})'.format(word)] = (word.lower() in document_words)
  return features

random.shuffle(documents)
featuresets = [(document_features(d), c) for (d,c) in documents]
# todo set this number
train_set, test_set = featuresets[:1000], featuresets[1000:]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# check accuracy
print(nltk.classify.accuracy(classifier, test_set))

# which words are informative?
classifier.show_most_informative_features(10)

# which examples did we get wrong?
errors = []
for (d, c) in documents:
  guess = classifier.classify(document_features(d))
  if guess != c:
    errors.append( (c, guess, d) )

for(c, guess, d) in sorted(errors):
  print 'correct=%-8s guess=%-8s doc=%-30s' % (c, guess, d)