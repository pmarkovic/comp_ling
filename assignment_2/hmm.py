import json


INITIAL_STATE = "initial_state"


class State:
    """
    """

    def __init__(self, name, initial_probs=None):
        self._name = name
        self._backpointer = None
        self._max_prob = None
        self._initial_probs = initial_probs

    def get_name(self):
        return self._name

    def get_backpointer(self):
        return self._backpointer

    def set_backpointer(self, value):
        self._backpointer = value

    def get_max_prob(self):
        return self._max_prob

    def set_max_prob(self, value):
        self._max_prob = value

    def get_initial_probs(self):
        return self._initial_probs

    def get_initial_prob(self, state):
        return self._initial_probs[state]


class HMM:
    """
    """

    def __init__(self, is_config, config_path):
        self._config_path = config_path
        self._trellis = list()
        self._tags = list()

        if is_config:
            self._read_config()
        else:
            # TODO implementation for training
            pass

    def _read_config(self):
        assert self._config_path.endswith(".json")

        with open(self._config_path) as config_file:
            configuration = json.load(config_file)

            self._states = configuration["states"]
            self._transitions = configuration["transitions"]
            self._emissions = configuration["emissions"]
            self._initial_state = State(INITIAL_STATE, configuration["initials"])

    def do_viterbi(self, sentence):
        self._trellis.clear()
        self._trellis.append(self._initial_state)

        for word in sentence:
            time_t_states = dict()

            for state in self._states:
                if type(self._trellis[-1]) is State:
                    time_t_states[state] = self._trellis[-1].get_initial_prob(state) * self._emissions[state][word]
                else:
                    prev_max_state = self._calc_prev_max()
                    time_t_states[state] = prev_max_state
                        

    def calc_prev_max(self, )


    def print_model(self):
        print(self._states)
        print(self._transitions)
        print(self._emissions)
        print(self._initials)
