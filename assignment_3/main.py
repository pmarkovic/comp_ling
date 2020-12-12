from parser import Parser

def main(args):
    print(args)
    
    parser = Parser()
    print(f"Does grammar recognize word: {parser.do_parsing('c')}")
    #parser.print_nodes()



if __name__ == "__main__":
    hello = "Let's welcome CKY!"

    main(hello)