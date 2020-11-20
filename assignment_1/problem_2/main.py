import time
import os
import nltk
from nltk.corpus import reuters

from ngram import BasicNgram, goodturing_estimator
from generator import Generator


def main():
    # Main thread of the program
    
    corpus = reuters.words()
    ngram = BasicNgram(2, corpus)
    generator = Generator(ngram)

    print(f"Text generated: {generator.generate(2, to_file=False)}")


if __name__ == "__main__":
    """
    Program starts here.
    """

    # To measure runtime
    start_time = time.time()

    corpus_dir = "/home/pavle/nltk_data/corpora/"
    
    # To check if corpus is already downloaded and to download it if it wasn't
    # nltk on my machine downloads corpora to nltk_data directory
    # Here is doc for download interface: https://www.nltk.org/_modules/nltk/downloader.html
    # This check can be omitted, just left nltk.download("reuters")
    if not os.path.isfile(os.path.join(corpus_dir, "reuters.zip")):
        nltk.download("reuters")
    
    main()
    print(f"Full program time: {time.time() - start_time} sec")
