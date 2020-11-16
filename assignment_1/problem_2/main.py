import nltk
from nltk.corpus import reuters

from ngram import BasicNgram


def main():
    corpus = reuters.words()
    bigram = BasicNgram(2, corpus)

    #print(bigram.contexts()[:50])

    incipts = bigram.incipts()
    #print(len(incipts))
    print(incipts.samples())

if __name__ == "__main__":
    #nltk.download("reuters")
    
    main()