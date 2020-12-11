import nltk


class Node:
    """
    """

    def __init__(self, name):
        self._name = name
        self._subtrees_total_count = 0
        self._terminals = dict()
        self._backpointers = dict()

    def get_name(self):
        return self._name

    def get_subtrees_total_count(self):
        return self._subtrees_total_count

    def update_subtrees_total_count(self, count):
        self._subtrees_total_count += count

    def get_terminals(self):
        return self._terminals

    def init_terminals(self, productions):
        self._terminals = productions

    def get_backpointers(self):
        return self._backpointers

    def is_empty_backpointers(self):
        return len(self._backpointers) == 0

    def __str__(self):
        return f"Node name: {self._name}\n \
                 Subtrees count: {self._subtrees_total_count}\n \
                 Terminals: {self._terminals}\n \
                 Backpointers: {self._backpointers}\n"


class Parser:
    """
    """

    def __init__(self, grammar_path="./grammars/my_grammar.cfg"):
        self._grammar = nltk.data.load(grammar_path)
        self._nodes = dict()

    def do_parsing(self, sentence="a a a a"):
        words = sentence.split(' ')
        sent_len = len(words)
        
        # Initialize nodes that correspodent to length 1 sentence sequences
        for index in range(sent_len):
            word = words[index]
            new_node = Node(self._create_node_name(index, index+1))

            new_node.init_terminals(self._grammar.productions(rhs=word))

            self._nodes[new_node.get_name()] = new_node

        # All other nodes that correspodent to lengths > 1 sentence sequences
        # Sequence length
        for length in range(2, sent_len + 1):
            # Start position of sequence
            for start_pos in range(sent_len - length + 1):
                new_node = Node(self._create_node_name(start_pos, start_pos+length))

                print(f"Node name: {new_node.get_name()}")
                # Dividing sequence into 2 subsequence
                for left_offset in range(1, length):
                    left_node = self._nodes[self._create_node_name(start_pos, start_pos+left_offset)]
                    right_node = self._nodes[self._create_node_name(start_pos+left_offset, start_pos+length)]
                    
                    # Go through terminals and backpointers
                    if left_node.is_empty_backpointers():
                        for terminal in left_node.get_terminals():
                            print(terminal.lhs())
                            print(self._grammar.productions(lhs=terminal.lhs()))

                self._nodes[new_node.get_name()] = new_node

                print()

    def _create_node_name(self, left, right):
        return f"node_{left}_{right}"

    def print_nodes(self):
        for node in self._nodes.values():
            print(f"{node}")


