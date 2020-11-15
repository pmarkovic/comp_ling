import nltk
from nltk.corpus import reuters

from problem_2.ngram import BasicNgram


def main():
    corpus = reuters.words()
    bigram = BasicNgram(2, corpus)

    print(bigram.contexts()[:50])
    print(bigram.generate())

if __name__ == "__main__":
    #nltk.download("reuters")
    
    print(main())