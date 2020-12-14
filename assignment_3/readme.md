Project file structure:
main.py
parser.py
util.py
outputs/result.txt
outputs/result_nltk.txt
outputs/parsed_trees.json
outputs/my_sentences.txt
outputs/tree_1.png
outputs/tree_2.png
outputs/tree_3.png
readme.md

Sentence for pictures:
can you tell me about the flights from saint petersbourg to toronto again .

Environment:
- python 3.8
- nltk 3.5
- manjaro 20.1

Runtimes:
- Grammar initialization runtime: ~1 sec
- Test set without generating parse trees: ~12 sec
- Test set with generating parse trees: ~652 sec
- A sentence with generating parse trees (5): ~1 sec
- A sentence with generating parse trees (36122): ~318 sec

How to run code:
usage: main.py [-h] [-gram_path GRAM_PATH] [-sents_path SENTS_PATH]
               [-result_file RESULT_FILE] [-trees_file TREES_FILE]
               [-sent SENT] [-draw]

An implementation of CYK algorithm for sentence parsing.

optional arguments:
  -h, --help            show this help message and exit
  -gram_path GRAM_PATH  File path of a grammar.
  -sents_path SENTS_PATH
                        File path of test sentences.
  -result_file RESULT_FILE
                        File where to write results of parsing test sentences.
  -trees_file TREES_FILE
                        File where to write all possible parsed trees for test
                        sentences.
  -sent SENT            Sentence to be parsed.
  -draw                 Flag to indicate if parse trees should be print or
                        drawn. Omit if don't want.

