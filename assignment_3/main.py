from parser import Parser

def main(args):
    print(args)

    parser = Parser()
    parser.do_parsing()
    #parser.print_nodes()


if __name__ == "__main__":
    hello = "Let's welcome CKY!"

    main(hello)