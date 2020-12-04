import json
from collections import deque
import numpy as np
from nltk.corpus.reader.conll import ConllCorpusReader


INITIAL_STATE = "initials"
STATES        = "states"
WORDS         = "words"
TRANSITIONS   = "transitions"
EMISSIONS     = "emissions"
UNK_CRUDE     = "crude"
UNK_MEAN      = "mean"
UNK_LOWEST    = "lowest"
END_TOKEN     = "<END>"


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

    def __init__(self, full_emissions, add_one, unk_words, end_token, config_path, data_path, save_model_path):
        assert config_path is not None or train_path is not None

        self._full_emissions = full_emissions
        self._add_one = add_one
        self._unk_words = unk_words
        self._end_token = end_token
        self._config_path = config_path
        self._data_path = data_path
        self._save_model_path = save_model_path

        self._trellis = Trellis()
        self._tags = deque()

        if self._config_path:
            self._read_config()
        else:
            self._train_model()

    def _read_config(self):
        assert self._config_path.endswith(".json")

        try:
            with open(self._config_path) as config_file:
                configuration = json.load(config_file)

                self._states = configuration[STATES]
                self._words = configuration[WORDS]
                self._transitions = configuration[TRANSITIONS]
                self._emissions = configuration[EMISSIONS]
                self._initial_state = State(INITIAL_STATE, max_prob=1.0)
        except FileNotFoundError:
            print("Not able to open config file!")
        

    def _train_model(self):
        self._states = list()
        self._words = list()
        self._transitions = {INITIAL_STATE: dict()}
        self._emissions = dict()

        corpus = ConllCorpusReader(self._data_path, ".tt", ["words", "pos"])
        sent_count = 0

        for sent in corpus.tagged_sents("de-train.tt"):
            sent_count += 1
            
            if self._end_token:
                sent.append((END_TOKEN, END_TOKEN))

            for i in range(len(sent) - 1):
                curr_tag = sent[i][1]

                if curr_tag not in self._transitions[INITIAL_STATE]:
                    self._init_config(curr_tag)
                
                if i == 0:
                    self._transitions[INITIAL_STATE][curr_tag] += 1

                if sent[i+1][1] not in self._transitions[INITIAL_STATE]:
                    self._init_config(sent[i+1][1])

                self._transitions[curr_tag][sent[i+1][1]] += 1

                self._update_emissions(curr_tag, sent[i][0])
            
            self._update_emissions(sent[-1][1], sent[-1][0])
        
        self._calc_probs(sent_count)

        self._initial_state = State(INITIAL_STATE, max_prob=1.0)

        if self._save_model_path:
            config = {STATES: self._states, \
                     WORDS: self._words, \
                     TRANSITIONS: self._transitions, \
                     EMISSIONS: self._emissions}
            with open(self._save_model_path, 'w') as json_file:
                json.dump(config, json_file, indent=4)

    def _init_config(self, tag):
        self._states.append(tag)

        for key in self._transitions.keys():
            self._transitions[key][tag] = 0
        
        self._transitions[tag] = {state: 0 for state in self._states}
        
        if self._full_emissions:
            self._emissions[tag] = {word: 0 for word in self._words}
        else:
            self._emissions[tag] = dict()

    def _update_emissions(self, tag, word):
        if word not in self._emissions[tag]:
            if self._full_emissions:
                self._words.append(word)
                for key in self._emissions.keys():
                    self._emissions[key][word] = 0
            else:
                self._emissions[tag][word] = 0

        self._emissions[tag][word] += 1

    def _calc_probs(self, sent_count):
        if self._add_one:
            num_of_states = len(self._states)
            num_of_words = len(self._words)

        # Calculating probabilities for transitions
        for key in self._transitions.keys():
            if key == INITIAL_STATE:
                for next_state in self._transitions[key].keys():
                    if self._add_one:
                        self._transitions[key][next_state] += 1
                        self._transitions[key][next_state] /= (sent_count + num_of_states)
                    else:
                        self._transitions[key][next_state] /= sent_count
                continue
            
            continues_count = sum(self._transitions[key].values())
            for next_state in self._transitions[key].keys():
                if self._add_one:
                    self._transitions[key][next_state] += 1
                    self._transitions[key][next_state] /= (continues_count + num_of_states)
                else:
                    self._transitions[key][next_state] /= continues_count

        # Calculating probabilities for emissions
        for key in self._emissions.keys():
            emissions_count = sum(self._emissions[key].values())
            for emssion in self._emissions[key]:
                if self._add_one:
                    self._emissions[key][emssion] += 1
                    self._emissions[key][emssion] /= (emissions_count + num_of_words)
                else:
                    self._emissions[key][emssion] /= emissions_count

    def test_model(self, test_path, save_test_file):
        corpus = ConllCorpusReader(test_path, ".t", ["words", "pos"])
        result = list()

        for sent in corpus.sents("de-test.t"):
            if self._end_token:
                sent.append(END_TOKEN)

            self.do_viterbi(sent)
            result.append(list(zip(sent, self._tags)))

        try:
            with open(save_test_file, 'w') as conll_file:
                for sent in result:
                    for pair in sent:
                        conll_file.write("\t".join(pair)+'\n')
                    conll_file.write('\n')
        except FileNotFoundError:
            print("Not able to open the file for test writing!")

            return False
        
        return True

    def do_viterbi(self, sentence):
        self._trellis.clear_model()
        self._trellis.add_timestep([self._initial_state])

        for timestep, word in enumerate(sentence):
            timestep_states = list()

            for state in self._states:
                backpointer = None
                curr_max_prob = -1

                for prev_state in self._trellis.get_timestep_states(timestep):
                    emission = 0.0

                    if word in self._emissions[state]:
                        emission = self._emissions[state][word] 
                    elif self._unk_words == UNK_CRUDE:
                        emission = 1.0
                    elif self._unk_words == UNK_MEAN:
                        emission = np.mean(list(self._emissions[state].values()))
                    elif self._unk_words == UNK_LOWEST:
                        emission = np.min(list(self._emissions[state].values()))

                    prob = prev_state.get_max_prob() \
                        * self._transitions[prev_state.get_name()][state] \
                        * emission

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
        print(f"Predicted tags: {' '.join(self._tags)}")

    def print_model(self):
        print("----- Model summary -----")
        print(f"Number of states: {len(self._states)}\nStates: {self._states}")
        print()
        print(f"Number of words: {len(self._words)}\nWords: {self._words}")
        print()
        print("Transition probabilities:")
        for key, value in self._transitions.items():
            print(f"{key}: {value}")
        print()
        print("Emission probabilities:")
        for key, value in self._emissions.items():
            print(f"{key}: {value}")

    def check_total_probs(self):
        print("Transitions:")
        for key, value in self._transitions.items():
            print(f"{key}: {sum(value.values())}")
        
        print("Emissions:")
        for key, value in self._emissions.items():
            print(f"{key}: {sum(value.values())}")
        
