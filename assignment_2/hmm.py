import json
from collections import deque 


INITIAL_STATE = "initials"


class State:
    """
    """

    def __init__(self, name, backpointer=None, max_prob=None):
        self._name = name
        self._backpointer = backpointer
        self._max_prob = max_prob

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


class Trellis:
    """
    """

    def __init__(self):
        self._model = list()
        self._last_state = None

    def add_timestep(self, timestep):
        self._model.append(timestep)

    def get_timestep_states(self, timestep):
        assert 0 <= timestep < len(self._model)

        return self._model[timestep]

    def get_model(self):
        return self._model

    def clear_model(self):
        self._model.clear()

    def get_last_state(self):
        return self._last_state

    def set_last_state(self):
        self._last_state = max(self._model[-1], key=lambda x: x.get_max_prob())



class HMM:
    """
    """

    def __init__(self, is_config, config_path):
        self._config_path = config_path
        self._trellis = Trellis()
        self._tags = deque()

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
            self._initial_state = State(INITIAL_STATE, max_prob=1.0)

    def do_viterbi(self, sentence):
        self._trellis.clear_model()
        self._trellis.add_timestep([self._initial_state])

        for timestep, word in enumerate(sentence):
            timestep_states = list()

            for state in self._states:
                backpointer = None
                curr_max_prob = -1

                for prev_state in self._trellis.get_timestep_states(timestep):
                    prob = prev_state.get_max_prob() * self._transitions[prev_state.get_name()][state] * self._emissions[state][word]

                    if prob > curr_max_prob:
                        curr_max_prob = prob
                        backpointer = prev_state.get_name()
                
                timestep_states.append(State(state, backpointer=backpointer, max_prob=curr_max_prob))

            self._trellis.add_timestep(timestep_states)
            self._trellis.set_last_state()

        self._do_tagging()

    def _do_tagging(self):
        self._tags.clear()
        curr_state = self._trellis.get_last_state()

        for timestep in list(reversed(self._trellis.get_model()[:-1])):
            self._tags.appendleft(curr_state.get_name())

            for state in timestep:
                if state.get_name() == curr_state.get_backpointer():
                    curr_state = state
                    break

    def print_tags(self):
        print(f"Tags: {' '.join(self._tags)}")

    def print_model(self):
        print(self._states)
        print(self._transitions)
        print(self._emissions)
