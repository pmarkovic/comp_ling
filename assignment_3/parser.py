import nltk
import json
from collections import deque

# For backpointers DS
SYMBOL_PRODS      = 1

# For collecting parse trees
LEFT_NODE_NAME    = 0
LEFT_NODE_SYMBOL  = 1
RIGHT_NODE_NAME   = 2
RIGHT_NODE_SYMBOL = 3


class Node:
    """
    Class represents a cell in a grid for the CKY algorithm.

    Nodes are used to store information about backpointers
    for every possible path from a node. Backpointers are then used 
    in parsing phase to generate all  possible trees from the start symbol. 
    It is dict data structure, KEY is nonterminal symbol of a grammar, 
    while VALUE is list.
    Elements of list for every KEY are:
        index 0 - number of all possible subtrees from a node using 
                  KEY symbol
        index 1 - list of tuples where a 4-element tuple represent 
                  pair of nodes which can be visited from the current node.
                  Order in a tuple is:
                    index 0 - left node that can be visited
                    index 1 - nonterminal symbol for the left node
                    index 2 - right node that can be visited
                    index 3 - nonterminal symbol for the right node
    Note: for leaf nodes, VALUE is not list but single terminal symbols.
    Example of backpointers data structure:
    {SIGMA: [3, [('node_0_1', VERB_MD, 'node_1_14', BJG)]]}

    Terminal attribute is a bool value to make distinction between 
    leaf and internal nodes.
    
    Subtrees_total_count attribute stores information how many 
    possible subtrees can be generated from a node. 
    """

    def __init__(self, name, terminal=False):
        self._name = name
        self._terminal = terminal
        self._subtrees_total_count = 0
        self._backpointers = None

    def get_name(self):
        return self._name

    def is_terminal(self):
        return self._terminal

    def get_subtrees_total_count(self):
        return self._subtrees_total_count

    def update_subtrees_total_count(self, count):
        self._subtrees_total_count += count

    def get_backpointers(self):
        return self._backpointers
    
    def set_backpointers(self, backpointers):
        self._backpointers = backpointers

    def is_empty_backpointers(self):
        return len(self._backpointers) == 0

    def get_productions(self, prod):
        return self._backpointers[prod]

    def get_production_count(self, prod):
        return self._backpointers[prod][0]

    def __str__(self):
        return f"""Node name: {self._name}\n \
                 Is terminal: {self._terminal}\n \
                 Subtrees count: {self._subtrees_total_count}\n \
                 Backpointers: {self._backpointers}\n"""


class Parser:
    """
    Main class responsible for the CKY algorithm.
    """

    def __init__(self, grammar_path):
        print("Loading grammar...")
        self._grammar = nltk.data.load(grammar_path)
        self._nodes = dict()

    def do_parsing(self, file_path, result_file, trees_file):
        """
        Wrapper method around parse_sentence method
        to do parsing for more sentences.
        ...

        Parameters:
        -----------
        file_path : str
            Path to a file where test sentences are.
        result_file : str
            Path to a file where to write results.
        trees_file : str
            Path to a file where to store parsed trees.
        """
        result = list()
        parsed_trees = dict()

        sents = nltk.data.load(file_path)
        test_sents = nltk.parse.util.extract_test_sentences(sents)

        for sent in test_sents:
            result.append(" ".join(sent[0]) + "\t" + str(self.parse_sentence(sent[0])))
            parsed_trees[" ".join(sent[0])] = self.generate_parse_tree(len(sent[0]))

        with open(result_file, 'w', encoding="utf-8") as writer:
            writer.write("\n".join(result))

        with open(trees_file, 'w') as json_file:
            json.dump(parsed_trees, json_file)

    def parse_sentence(self, sentence):
        """
        Method to do recognizer part, but also to store backpointers for every node.
        Additionally, number of subtrees that can be generated from every node is calculated
        along the process. Therefore, at the end, the last node (node_0_[sent_len])
        will contain number of all possible parse trees.
        ...

        Parameters:
        -----------
        sentence : list
            Tokenized sentence to be processed.

        Return:
        -------
        int
            Number of all possible parse trees.
        """
        sent_len = len(sentence)
        # Check for empty sentence
        if len(sentence) == 0:
            return False

        self._nodes.clear()
        
        # Initialize nodes that correspodent to length 1 sentence sequences
        for index in range(sent_len):
            word = sentence[index]
            new_node = Node(self._create_node_name(index, index+1), terminal=True)

            # Special case if length is just one word.
            if sent_len == 1:
                start = self._grammar.start()
                new_node.set_backpointers({start: \
                                            (1, self._grammar.productions(lhs=start, rhs=word))})
            else:
                new_node.set_backpointers({prod.lhs(): \
                                          (1, prod.rhs()[0]) \
                                            for prod in self._grammar.productions(rhs=word)})
            new_node.update_subtrees_total_count(len(new_node.get_backpointers()))
            self._nodes[new_node.get_name()] = new_node

        # CKY algorithm
        # All other nodes that correspodent to lengths > 1 sentence sequences
        # Sequence length
        for length in range(2, sent_len + 1):
            # Start position of sequence
            for start_pos in range(sent_len - length + 1):
                new_node = Node(self._create_node_name(start_pos, start_pos+length))
                backpointers = dict()

                # Dividing sequence into 2 subsequence
                for left_offset in range(1, length):
                    left_node = self._nodes[self._create_node_name(start_pos, start_pos+left_offset)]
                    right_node = self._nodes[self._create_node_name(start_pos+left_offset, start_pos+length)]
                    
                    # Find all possible productions from the current node to
                    # left and right nodes among all in grammar
                    # such that rhs is -> ln_lhs prod.rhs()[1].
                    # ln_lhs is a KEY in left_node backpointers (nonterminal symbol)
                    # prod.rhs()[1] is a right symbol on rhs of production which has ln_lhs as left symbol
                    for ln_lhs in left_node.get_backpointers():
                        for prod in self._grammar.productions(rhs=ln_lhs):
                            if prod.rhs()[1] in right_node.get_backpointers():

                                # For the last node (node_0_[sent_len]) only productions with SIGMA on lhs
                                if length == sent_len and prod.lhs() != self._grammar.start():
                                    continue

                                new_node_prod = prod.lhs()
                                prod_nodes = (left_node.get_name(), ln_lhs, right_node.get_name(), prod.rhs()[1])
                                # Calculate number of all possible subtrees from the current node
                                # using the current production
                                prod_count = left_node.get_production_count(ln_lhs) \
                                            * right_node.get_production_count(prod.rhs()[1])
                                
                                if not new_node_prod in backpointers:
                                    backpointers[new_node_prod] = [0, list()]
                                backpointers[new_node_prod][0] += prod_count
                                backpointers[new_node_prod][1].append(prod_nodes)

                                new_node.update_subtrees_total_count(prod_count)

                new_node.set_backpointers(backpointers)
                self._nodes[new_node.get_name()] = new_node

        return self._nodes[self._create_node_name(0, sent_len)].get_subtrees_total_count()

    def generate_parse_tree(self, n, draw=False):
        """
        Method to generate a parse trees using backpointers from nodes.
        ...

        Parameters:
        -----------
        n : int
            Length of parsed sentence.
        draw : bool (optional)
            Flag to indicate if trees should be drawn or print on stdout.
        
        Return:
        -------
        trees : list
            List of all possible parse trees.
        """
        if len(self._nodes) == 0:
            print("The parsing step is required first. There are no nodes!")
            return False

        # Generate all possible parse trees
        trees = self._collect_trees(self._create_node_name(0, n), self._grammar.start())

        if not trees:
            print("Oh, there are no trees.")
        
        # Process trees to be in the right form
        trees = [str(tree).replace(',', '') for tree in trees]

        # Create tree objects from strings
        trees = [nltk.Tree.fromstring(tree, brackets="[]") for tree in trees]

        for tree in trees:
            if draw:
                tree.draw()
            else:
                tree.pretty_print()

        return trees


    def _create_node_name(self, left, right):
        return f"node_{left}_{right}"

    def _collect_trees(self, node_name, symbol):
        """
        Main method for generating parse trees.
        Recursively going from the last node (node_0_[sent_len]) and
        follow backpointers to leaves.
        ...

        Parameters:
        -----------
        node_name : str
            Name of the current node.
        symbol : nonterminal
            Nonterminal symbol of grammar.

        Return:
        -------
        curr_trees : list
            List of possible parse trees.
        """
        node = self._nodes[node_name]

        if node.get_subtrees_total_count() == 0:
            return list()

        # Base case
        if node.is_terminal():
            return [node.get_productions(symbol)[SYMBOL_PRODS]]

        curr_trees = list()
        for prod in node.get_productions(symbol)[SYMBOL_PRODS]:
            # Go to left subtrees
            left_trees = self._collect_trees(prod[LEFT_NODE_NAME], prod[LEFT_NODE_SYMBOL])
            # Go to right subtrees
            right_trees = self._collect_trees(prod[RIGHT_NODE_NAME], prod[RIGHT_NODE_SYMBOL])

            # Combine left and right subtrees
            for left_tree in left_trees:
                for right_tree in right_trees:
                    curr_trees.append([symbol,
                                      [prod[LEFT_NODE_SYMBOL], left_tree],
                                      [prod[RIGHT_NODE_SYMBOL], right_tree]])

        return curr_trees

    def print_nodes(self, to_file=False):
        """
        Helper method used to check if nodes are correct.
        ...

        Parameters:
        -----------
        to_file : bool (optional)
            Flag to indicate if nodes should be written in a file
        """
        nodes = "\n".join([str(node) for node in self._nodes.values()])

        if to_file:
            with open("./outputs/nodes.txt", 'a') as writer:
                writer.write(nodes)
        else:
            print(nodes)

