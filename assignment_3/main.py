from parser import Parser
from util import test_nltk_parser, compare_results


def main(args):
    print(args)

    #check_nltk_sent("i 'd like to fly from buffalo to either orlando or long beach .".split(' '))

    parser = Parser("./grammars/atis-grammar-cnf.cfg")
    
    #parser.do_parsing("./grammars/atis-test-sentences.txt")

    sentence = "prices .".split(' ')
    print(parser.parse_sentence(sentence))

    parse_trees = parser.generate_parse_tree(len(sentence))
    if parse_trees:
        for tree in parse_trees:
            tree.draw()
        print(f"Number of parse trees: {len(parse_trees)}")
    else:
        print(f"Oh, there are no any tree!")
    #parser.print_nodes(True)

    #test_nltk_parser()
    compare_results()

if __name__ == "__main__":
    hello = "Let's welcome CKY!"

    main(hello)