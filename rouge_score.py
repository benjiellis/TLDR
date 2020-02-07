import tldr
from pythonrouge.pythonrouge import Pythonrouge
# need to import document set and gold rouge scores

def calculate_rouge(test_summary, gold_summary, ngram_size=2):
    # calculates rouge score for one document and its suggested summary
    # input: document as cleaned plaintext string and summary as plaintext string
    # output: rouge score of how well summary is a summary of document
    test_summary = [test_summary]
    gold_summary = [[gold_summary]]

    #print("test_summary", test_summary)
    #print("gold_summary", gold_summary)
    rouge = Pythonrouge(summary_file_exist=False, summary=test_summary, reference=gold_summary,
                        n_gram=ngram_size, ROUGE_SU4=False, ROUGE_L=True, stemming=True, recall_only=True,
                        stopwords=True, word_level=True, length_limit=True, length=50,use_cf=False,
                        cf=95, scoring_formula='average', resampling=True, samples=1000, favor=True, p=0.5)
    return rouge.calc_score()
