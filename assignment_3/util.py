"""
Helper functions used to test this implementation with nltk parser.
IMPORTANT: This functions can not be called through command line args.
           Used only for testing purpose.
"""

def test_nltk_parser():
    """
    The function that parse test sentences using nltk parser.
    It stores result in ./outputs/result_nltk.txt
    """
    result = list()
    grammar = nltk.data.load("./grammars/atis-grammar-cnf.cfg")
    parser = nltk.parse.BottomUpChartParser(grammar)

    sents = nltk.data.load("./grammars/atis-test-sentences.txt")
    test_sents = nltk.parse.util.extract_test_sentences(sents)

    counter = 0

    for sent in test_sents:
        # Naive progress bar
        if counter % 10 == 0:
            print(f"Processed {counter} sents...")
        counter += 1

        # Number of parsed trees, will be set to 0
        # if a terminal is not in grammar
        parses_count = -1
        try:
            chart = parser.chart_parse(sent[0])
        except Exception as er:
            parses_count = 0

        if parses_count == 0:
            result.append(" ".join(sent[0]) + f"\t{parses_count}")
        else:
            parses_count = len(list(chart.parses(grammar.start())))
            result.append(" ".join(sent[0]) + f"\t{parses_count}")
    
    with open("./outputs/result_nltk.txt", 'w') as writer:
        writer.write("\n".join(result))


def compare_results():
    """
    Function used to test results between this implementation and nltk.
    It just goes line by line in both result files 
    and checks if the numbers match.
    It prints sentences if numbers don't match.
    """
    with open("./outputs/result.txt", 'r') as my_result:
        my_lines = my_result.readlines()

    with open("./outputs/result_nltk.txt", 'r') as nltk_result:
        nltk_lines = nltk_result.readlines()

    for i in range(len(my_lines)):
        my_line = my_lines[i].split('\t')
        nltk_line = nltk_lines[i].split('\t')

        if my_line[1] != nltk_line[1]:
            print(f"{i}. {my_line[0]}")
            print(f"my count: {my_line[1]}, nltk count: {nltk_line[1]}")