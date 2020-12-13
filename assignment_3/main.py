import argparse
import time

from parser import Parser
from util import test_nltk_parser, compare_results


WELCOME_MSG = "Let's welcome CKY!"

def main(args):
    print(WELCOME_MSG)

    init_grammar_time = time.time()
    parser = Parser(args.gram_path)
    init_grammar_time = time.time() - init_grammar_time

    print("Start parsing...")
    parsing_time = time.time()
    
    # If sentence argument is provided then only the sentence will be parsed
    if args.sent:
        sentence = args.sent.split(' ')
        print(parser.parse_sentence(sentence))
        parser.generate_parse_tree(len(sentence), args.draw)
    else:
        parser.do_parsing(args.sents_path, args.result_file)

    parsing_time = time.time() - parsing_time
    print("Finish parsing!")

    print(f"Grammar initialization runtime: {init_grammar_time}")
    print(f"Parsing runtime: {parsing_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An implementation of CYK algorithm for sentence parsing.")
    parser.add_argument("-gram_path", default="./grammars/atis-grammar-cnf.cfg", type=str, help="File path of a grammar.")
    parser.add_argument("-sents_path", default="./grammars/atis-test-sentences.txt", type=str, help="File path of test sentences.")
    parser.add_argument("-result_file", default="./outputs/result.txt", type=str, help="File where to write results of parsing test sentences.")
    parser.add_argument("-sent", default=None, type=str, help="Sentence to be parsed.")
    parser.add_argument("-draw", default=False, action='store_true', help="Flag to indicate if parse trees should be print or draw. Omit if don't want.")
    args = parser.parse_args()

    start_time = time.time()
    print("Program started...")
    main(args)
    print(f"Program ended. Runtime: {time.time() - start_time} sec")
