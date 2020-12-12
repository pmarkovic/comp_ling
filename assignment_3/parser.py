import nltk


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
        self._grammar = nltk.data.load(grammar_path)
        self._nodes = dict()

    def do_parsing(self, sentence="a a a a"):
        # Check for empty sentence
        if not sentence:
            return False

        words = sentence.split(' ')
        sent_len = len(words)
        
        # Initialize nodes that correspodent to length 1 sentence sequences
        for index in range(sent_len):
            word = words[index]
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

        print(self._nodes[self._create_node_name(0, sent_len)])

        return self._nodes[self._create_node_name(0, sent_len)].get_subtrees_total_count() != 0

    def _create_node_name(self, left, right):
        return f"node_{left}_{right}"

    def print_nodes(self):
        for node in self._nodes.values():
            print(f"{node}")


