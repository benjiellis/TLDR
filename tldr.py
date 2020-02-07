import heapq, class_tagger, sys, re, rouge_score, load_docs, decimal
from bs4 import BeautifulSoup
from urllib import request
from nltk import word_tokenize as tokenize
from nltk.stem import SnowballStemmer
import spacy
from math import log

# TO DO:
# log frequency of words (Zipfian Distribution of words)
# changing slope of decline for sentence weights in inverted pyramid
# title information

sent_stemmer = spacy.load('en')

def spacy_sent_tokenize(article):
    #split the article into sentences
    tokens = sent_stemmer(article)
    sentences = []

    for sentence in tokens.sents:
        sentences.append(str(sentence))

    return sentences

def get_sentences_from_url(url):
    # code to extract plaintext sentences from a URL
    # input: a URL  as a string
    # note: this only works with NYTimes URLs
    # output: a list with each item as a sentence

    source = request.urlopen(url).read().decode("utf-8")
    soup = BeautifulSoup(source, "html.parser")
    paragraphs = soup.find_all('p', attrs={'class' : 'story-body-text story-content'})

    article = ""
    for paragraph in paragraphs:
        article += (paragraph.getText())
        article += " "

    return spacy_sent_tokenize(article)

def eval_baseline():
    baseline_1 = 0
    baseline_2 = 0
    baseline_L = 0
    scores = [0 for item in test_data]
    for index, document in enumerate(test_data):
        if(len(document) > 1):
            article = spacy_sent_tokenize(document[0])
            gold_summary = document[1]

            baseline_summary = article[0:5]
            print(baseline_summary)

            #compare with first n sentences as summary
            baseline_score = rouge_score.calculate_rouge(baseline_summary, gold_summary)
            baseline_1 += baseline_score["ROUGE-1"]
            baseline_2 += baseline_score["ROUGE-2"]
            baseline_L += baseline_score["ROUGE-L"]

            scores[index] = (baseline_score["ROUGE-2"] + baseline_score["ROUGE-L"]) / 2

    baseline_1 /= len(test_data)
    baseline_2 /= len(test_data)
    baseline_L /= len(test_data)

    print("Baseline ROUGE-1", round(baseline_1, 5))
    print("Baseline ROUGE-2", round(baseline_2, 5))
    print("Baseline ROUGE-L", round(baseline_L, 5))

    return (((baseline_2 + baseline_L)/2), scores)



def evaluate(test_data, stem_tokens=False, case_normalize=False, open_class_tagging=False, inverted_pyramid=False,
             log_weight=False, alpha=1):
    # this will test the summarizer against gold-labeled test data
    # and will return the accuracy of the summarizer
    # Input: test data in the form of (article, highlights) tuples
    # Note: do not separate article into sents

    # rouge score is a comparison metric (almost like Longest Common Subsequence)

    average = 0
    average_1 = 0
    average_2 = 0
    average_L = 0

    scores = [0 for item in test_data]
    for index, document in enumerate(test_data):
        if(len(document) > 1):
            article = spacy_sent_tokenize(document[0])
            gold_summary = document[1]

            summarizer = Summarizer(article,
                                    stem_tokens=stem_tokens,
                                    case_normalize=case_normalize,
                                    open_class_tagging=open_class_tagging,
                                    inverted_pyramid=inverted_pyramid,
                                    log_weight=log_weight,
                                    alpha=alpha)

            generated_summary = summarizer.get_summary()

            # The ROUGE score will work over multiple highlights in the gold summary
            score = rouge_score.calculate_rouge(generated_summary, gold_summary)

            scores[index] = (score["ROUGE-2"] + score["ROUGE-L"]) / 2
            average += scores[index]
            average_1 += score["ROUGE-1"]
            average_2 += score["ROUGE-2"]
            average_L += score["ROUGE-L"]
            #print(score, total)
        else:
            print("ERROR")

    # we want to maximize total, higher rouge score means greater content overlap
    # rouge-2 measures bigram similarity, and rouge-L measures subsequence similarity
    # summing gives us a good idea how similar they are
    average /= len(test_data)
    average_1 /= len(test_data)
    average_2 /= len(test_data)
    average_L /= len(test_data)

    print("ROUGE-1", round(average_1, 5))
    print("ROUGE-2", round(average_2, 5))
    print("ROUGE-L", round(average_L, 5))
    return (average, scores)

class Summarizer: # container object for summary data/weight model

    def __init__(self,
                 article,
                 stem_tokens=False,
                 case_normalize=False,
                 open_class_tagging=False,
                 inverted_pyramid=False,
                 log_weight=False,
                 alpha=1):
        self.sentences = article
        self.stems = stem_tokens
        self.no_cases = case_normalize
        self.open_class = open_class_tagging
        self.pyramid = inverted_pyramid
        self.word_weights = self.get_word_weights()
        self.log = log_weight
        self.alpha = alpha

    def get_word_weights(self):
        # code to create model of word weights
        # input: list of sentences, each sentence a string
        # output: dictionary with words as keys and weights as values
        weights = {}

        if(self.stems):
            stemmer = SnowballStemmer('english')

        # assume bag of words model for now, i.e., each instance increases word weight by 1
        for sentence in self.sentences:
            tokens = tokenize(sentence)
            # generate new list that only has words that are open class
            if self.open_class:
                tokens = [token for token in tokens if class_tagger.is_open_class(token)]

            if self.stems:
                tokens = [stemmer.stem(token) for token in tokens]

            else: # lemmas are already case normalized, only run this if not using lemmas
                if self.no_cases:
                    tokens = [token.lower() for token in tokens]

            if self.log:
                w = log(1)
            else:
                w = 1

            for token in tokens:
                if token not in weights:
                    weights[token] = w
                else:
                    weights[token] += w

        return weights

    def get_summary(self, n=5):
        # returns top weighted sentences in article, in order
        # input: integer n as number of sentences to return
        # output: list of sentences with top scores, in order

        # parallel list to list of sentences containing scores for those sentences
        scores = [0 for item in self.sentences]

        # calculate scores
        length = len(self.sentences)
        for index, sentence in enumerate(self.sentences):
            tokens = tokenize(sentence)
            for token in tokens:
                if token in self.word_weights:
                    #score += self.word_weights[token]
                    scores[index] += self.word_weights[token]
            if self.pyramid:
                # linearly decrease the weight of the sentence
                # based on its position in the article
                # pyramid_score = 1- (index / length)
                # score *= pyramid_score
                scores[index] *= (self.alpha - (self.alpha*index / length))

                # new formula, behaves the same when alpha = 1
                # increasing alpha increases the skew of weight to the front
                # alpha = 3 will have the first sentence three times larger, and last sentence at zero

        # take top n sentences
        top_indices = heapq.nlargest(n, range(len(scores)), scores.__getitem__)
            # ^^ python magic, returns indices of max n items
        top_indices.sort() # sort so they appear in correct order

        # create list of sentences from top indices and return it
        top_sentences = []
        for index in top_indices:
            top_sentences.append(self.sentences[index])

        return top_sentences

# The main function
if __name__ == "__main__":
    # Read in the CNN corpus
    # Call eval on the data using different features
    #test_data = load_docs.load_docs("cnn/stories/")
    #test_data = load_docs.load_docs("cnn/test/")
    #test_data = load_docs.load_docs("cnn/test2/")
    test_data = load_docs.load_docs("cnn/test/")

    baseline_eval, baseline_scores = eval_baseline()
    print("Baseline Score: ", round(baseline_eval, 5))

    standard_eval, scores = evaluate(test_data)
    print("Standard Algorithm Score: ", round(standard_eval, 5))

    stem_eval, stem_scores = evaluate(test_data, stem_tokens=True)
    print("Stemming Score: ", round(stem_eval, 5))

    case_normalized_eval, case_normalized_scores = evaluate(test_data, case_normalize=True)
    print("Case Normalization Score: ", round(case_normalized_eval, 5))

    open_class_eval, open_class_scores = evaluate(test_data, open_class_tagging=True)
    print("Open Class Tagging Score: ", round(open_class_eval, 5))

    inverted_pyramid_eval, inverted_pyramid_scores = evaluate(test_data, inverted_pyramid=True)
    print("Inverted Pyramid Score: ", round(inverted_pyramid_eval, 5))

    all_feats_eval, all_feats_scores = evaluate(test_data, stem_tokens=True, case_normalize=True, open_class_tagging=True, inverted_pyramid=True)
    print("All Features Score: ", round(all_feats_eval, 5))

    best_feats_eval, best_feats_scores = evaluate(test_data, open_class_tagging=True, inverted_pyramid=True)
    print("Best Features Score: ", round(best_feats_eval, 5))

    # I am expecting this function to be called from the command line like so:
    # python tldr.py https://www.nytimes.com/2017/11/21/technology/fcc-net-neutrality.html

    # Including support for a list of urls like so:
    # python tldr.py <URL>+

    # Note: this only works with NYTimes URLs"""

    #print(rouge_score.calculate_rouge(['The', 'quick', 'brown', 'fox'], ['The', 'slow', 'green', 'fox']))

    if len(sys.argv) > 1:
        urls = sys.argv[1:]

        for url in urls:
            #get the article text from the url
            article = get_sentences_from_url(url)
            #print(article)

            #summary = Summarizer(article).get_summary()
            #print(summary)

            # the modifiers to the Summarizer init :
            #stem_tokens=False, case_normalize=False, open_class_tagging=False, inverted_pyramid = False

            #pyramid_summary = Summarizer(article, inverted_pyramid=True).get_summary()
            #print(pyramid_summary)

            #stem_summary = Summarizer(article, stem_tokens=True).get_summary()
            #print(stem_summary)

            #open_class_summary = Summarizer(article, open_class_tagging=True).get_summary()
            #print(open_class_summary)

            all_feats_summary = Summarizer(article, stem_tokens=True, case_normalize=True, open_class_tagging=True, inverted_pyramid=True).get_summary(n=5)
            print(all_feats_summary)
