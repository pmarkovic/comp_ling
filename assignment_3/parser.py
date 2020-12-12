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
        words = sentence.split(' ')
        sent_len = len(words)
        
        # Initialize nodes that correspodent to length 1 sentence sequences
        for index in range(sent_len):
            word = words[index]
            new_node = Node(self._create_node_name(index, index+1), terminal=True)
            new_node.set_backpointers({prod.lhs(): prod.rhs()[0] for prod in self._grammar.productions(rhs=word)})
            self._nodes[new_node.get_name()] = new_node

    def _create_node_name(self, left, right):
        return f"node_{left}_{right}"

    def print_nodes(self):
        for node in self._nodes.values():
            print(f"{node}")


