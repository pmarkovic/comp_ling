from parser import Parser, test_nltk_parser

def main(args):
    print(args)
    
    parser = Parser("./grammars/atis-grammar-cnf.cfg")
    
    parser.do_parsing("./grammars/atis-test-sentences.txt")

    #parser.parse_sentence("list round trips .")
    #parser.print_nodes(True)

    #test_nltk_parser()

if __name__ == "__main__":
    hello = "Let's welcome CKY!"

    main(hello)