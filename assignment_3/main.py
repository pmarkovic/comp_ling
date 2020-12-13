from parser import Parser
from util import test_nltk_parser, compare_results


def main(args):
    print(args)

    parser = Parser("./grammars/atis-grammar-cnf.cfg")

    sentence = "can you tell me about the flights from saint petersburg to toronto again .".split(' ')
    print(parser.parse_sentence(sentence))

    parse_trees = parser.generate_parse_tree(len(sentence), True)


if __name__ == "__main__":
    hello = "Let's welcome CKY!"

    main(hello)