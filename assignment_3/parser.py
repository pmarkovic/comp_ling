import nltk
from collections import deque


class Node:
    """
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
    """

    def __init__(self, grammar_path="./grammars/my_grammar.cfg"):
        print("Loading grammar...")
        self._grammar = nltk.data.load(grammar_path)
        self._nodes = dict()

    def do_parsing(self, file_path):
        print("Start parsing...")
        result = list()

        sents = nltk.data.load(file_path)
        test_sents = nltk.parse.util.extract_test_sentences(sents)

        for sent in test_sents:
            result.append(" ".join(sent[0]) + "\t" + str(self.parse_sentence(sent[0])))

        with open("result2.txt", 'w', encoding="utf-8") as writer:
            writer.write("\n".join(result))

        print("Finish parsing!")

    def parse_sentence(self, sentence):
        sent_len = len(sentence)
        # Check for empty sentence
        if len(sentence) == 0:
            return False

        self._nodes.clear()
        
        # Initialize nodes that correspodent to length 1 sentence sequences
        for index in range(sent_len):
            word = sentence[index]
            new_node = Node(self._create_node_name(index, index+1), terminal=True)
            new_node.set_backpointers({prod.lhs(): (1, prod.rhs()[0]) for prod in self._grammar.productions(rhs=word)})
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
                    
                    for ln_lhs in left_node.get_backpointers():
                        for prod in self._grammar.productions(rhs=ln_lhs):
                            if prod.rhs()[1] in right_node.get_backpointers():

                                # For the last node only productions with SIGMA on lhs
                                if length == sent_len and prod.lhs() != self._grammar.start():
                                    continue

                                new_node_prod = prod.lhs()
                                prod_nodes = (left_node.get_name(), ln_lhs, right_node.get_name(), prod.rhs()[1])
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

    def generate_parse_tree(self, n):
        if len(self._nodes) == 0:
            print("The parsing step is required first. There are no nodes!")
            return False

        trees = self._collect_trees(self._create_node_name(0, n), self._grammar.start())
        trees = [str(tree).replace(',', '') for tree in trees]
        trees = [nltk.Tree.fromstring(tree, brackets="[]") for tree in trees]

        return trees


    def _create_node_name(self, left, right):
        return f"node_{left}_{right}"

    def _collect_trees(self, node_name, symbol):
        node = self._nodes[node_name]

        if node.is_terminal():
            return [node.get_productions(symbol)[1]]

        curr_trees = list()
        for prod in node.get_productions(symbol)[1]:
            left_trees = self._collect_trees(prod[0], prod[1])
            right_trees = self._collect_trees(prod[2], prod[3])

            for left_tree in left_trees:
                for right_tree in right_trees:
                    curr_trees.append([symbol, [prod[1], left_tree], [prod[3], right_tree]])

        return curr_trees

    def print_nodes(self, to_file=False):
        nodes = "\n".join([str(node) for node in self._nodes.values()])
        if to_file:
            with open("output.txt", 'a') as writer:
                writer.write(nodes)
        else:
            print(nodes)

