import time
import nltk
from nltk.corpus import reuters

from ngram import BasicNgram
from generator import Generator


def main():
    corpus = reuters.words()
    ngram = BasicNgram(2, corpus)
    generator = Generator(ngram)

    print(f"Text generated: {generator.generate(10, to_file=False)}")


if __name__ == "__main__":
    #nltk.download("reuters")
    
    start_time = time.time()
    main()
    print(f"Full program time: {time.time() - start_time} sec")