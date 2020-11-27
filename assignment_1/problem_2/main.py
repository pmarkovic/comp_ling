import time
import os
import argparse
import nltk
from nltk.corpus import reuters

from ngram import BasicNgram, goodturing_estimator
from generator import Generator


def main(args):
    # Main thread of the program
    
    corpus = reuters.words()
    print("Training ngram model...")
    ngram = BasicNgram(args.ngram, corpus)
    print(f"First 10 contexts: {ngram.contexts()[:10]}")
    print("Generating texts...")
    generator = Generator(ngram, args.textdir)

    print(f"Text generated: {generator.generate(args.ntexts,args.nwords, args.tofile, args.filename)}")


if __name__ == "__main__":
    """
    Program starts here.
    """

    # Parameters parsing
    parser = argparse.ArgumentParser(description="Generating news texts using a ngram model.")
    parser.add_argument("ngram", metavar='n', type=int, help="N for ngram model")
    parser.add_argument("-corpath", default="default_path", type=str, help="Path to corpora directory")
    parser.add_argument("-textdir", default= "", type=str, help="Path to directory where to write generated text")
    parser.add_argument("-ntexts", default=10, type=int, help="Number of texts to be generated")
    parser.add_argument("-nwords", default=100, type=int, help="Number of words per text to be generated")
    parser.add_argument("-tofile", default=False, type=bool, help="Flag to write texts into a file")
    parser.add_argument("-filename", default="generated_text.txt", type=str, help="Name of a file for generated text")
    args = parser.parse_args()

    # To measure runtime
    start_time = time.time()
    print("Program started...")
    
    # To check if corpus is already downloaded and to download it if it wasn't
    # nltk on my machine downloads corpora to nltk_data directory
    # Here is doc for download interface: https://www.nltk.org/_modules/nltk/downloader.html
    # This check can be omitted, just left nltk.download("reuters")
    if not os.path.isfile(os.path.join(args.corpath, "reuters.zip")):
        nltk.download("reuters")
    
    main(args)
    print(f"Program ended. Runtime: {time.time() - start_time} sec")
