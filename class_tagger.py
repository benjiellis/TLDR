# create a tagger function that decides if a word is open or closed class
# should just return a boolean

# making use of the set of closed class words being finite and relatively small,
# we can use a set to check existence in constant time
# list of closed class words taken from https://mailman.uib.no/public/corpora/2011-November/014318.html
list_of_closed_class_words = []

with open("closed_class.txt") as in_file:
    for line in in_file:
        clean = line.strip("\n")
        words = clean.split(" ")
        list_of_closed_class_words.extend(words)

set_of_closed_class_words = set(list_of_closed_class_words)

def is_open_class(word):
    return not word in set_of_closed_class_words