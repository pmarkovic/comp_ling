import time
import nltk
from nltk.corpus import reuters

from ngram import BasicNgram
from generator import Generator


def main(start_time):
    corpus = reuters.words()
    ngram = BasicNgram(1, corpus)
    generator = Generator(ngram)


if __name__ == "__main__":
    #nltk.download("reuters")
    
    start_time = time.time()
    main(start_time)
    print(f"Full program time: {time.time() - start_time} sec")