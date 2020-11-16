import nltk
from nltk.corpus import reuters

from ngram import BasicNgram
from generator import Generator


def main():
    corpus = reuters.words()
    bigram = BasicNgram(2, corpus)
    generator = Generator(bigram)
    
    print(generator.get_incipts()[20:30])

    print(f"Text is generated: {generator.generate(5, file_name="bigram_model_texts.txt")}")


if __name__ == "__main__":
    #nltk.download("reuters")
    
    main()