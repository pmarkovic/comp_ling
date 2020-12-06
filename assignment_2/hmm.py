import json
from collections import deque
import numpy as np
from nltk.corpus.reader.conll import ConllCorpusReader


INITIAL_STATE = "initials"
STATES        = "states"
WORDS         = "words"
TRANSITIONS   = "transitions"
EMISSIONS     = "emissions"
END_TOKEN     = "<END>"


class State:
    """
    Class to represent a state in trellis.
    """

    def __init__(self, name, backpointer=None, max_prob=None):
        """
        Class constructor.
        ...

        Parameters:
        -----------
        name : str
            Unique name of the state, for POS tagger the name of particular tag.
        backpointer : str
            To store the name of the previous state with max prob during unfolding trellis.
        max_prob : float
            Max probability to end up in a state.
        """
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
    Class to represent a trellis.
    """

    def __init__(self):
        """
        Class constructor
        ...
        
        Attributes:
        -----------
        model : list
            List of lists to store timesteps of trellis, 
            where one timestep is a list of states.
        last_state : State
            A state in the last timestep with the highest probability
        """
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
    Class that represents the Hiddent Markov Model.
    Model can be import from an existing config file, or
    trained from scratch by providing path to data.
    In the former case, config_path has to be provided, 
    in the later case, data_path has to be providede.

    When a model is setup or trained, it can be tested on a
    single sentence by calling do_viterbi, or on a test set
    by calling test_model. During testing a trellis object
    is used to store information about states and paths, while
    final tags will be stored in tags attribute. 
    """

    def __init__(self, full_emissions, add_one, end_token, \
                 config_path, data_path, save_model_path):
        """
        Class constructor
        ...

        Parameters:
        -----------
        full_emissions : bool
            Flag to indicates if full emissions table will be created,
            or if False emission probabilities only for seen words by tags will be created.
        add_one : bool
            Flag to indicates if add_one smoothing will be used
        end_token : bool
            Flag to indicates if <END> token will be used
        config_path : str
            Path to the config to use for setting up the model
        data_path : str
            Path to the train set
        save_model_path : str
            Path to the file to save trained model to
        """
        # One of paths (config_path and data_path) must be provided
        # If both are provided, the config_path will be used
        assert config_path is not None or data_path is not None

        self._full_emissions = full_emissions
        self._add_one = add_one
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
        """
        Method to setup the model from a config file (.json file).
        """
        assert self._config_path.endswith(".json")

        try:
            with open(self._config_path) as config_file:
                configuration = json.load(config_file)

                # List of all possible states (tags)
                self._states = configuration[STATES]
                # List of all words encountered during training
                self._words = configuration[WORDS]
                # Dictionary of dictionaries to store all transition probabilities
                # including transitions from initial state
                self._transitions = configuration[TRANSITIONS]
                # Dictionary of dictionaries to store all emission probabilities
                self._emissions = configuration[EMISSIONS]
                # Initial state from which model starts
                self._initial_state = State(INITIAL_STATE, max_prob=1.0)
        except FileNotFoundError:
            print("Not able to open config file!")
        

    def _train_model(self):
        """
        Method to train the model from scratch.
        """
        self._states = list()
        self._words = list()
        self._transitions = {INITIAL_STATE: dict()}
        self._emissions = dict()

        corpus = ConllCorpusReader(self._data_path, ".tt", ["words", "pos"])
        sent_count = 0

        # Processing sentence by sentence
        for sent in corpus.tagged_sents("de-train.tt"):
            sent_count += 1
            
            # Append end token if required
            if self._end_token:
                sent.append((END_TOKEN, END_TOKEN))

            # Process (word, tag) pair one by one bar the last one
            # because there is no transition after it, only emission
            for i in range(len(sent) - 1):
                curr_tag = sent[i][1]

                # Update the model if it the first time to see current tag
                if curr_tag not in self._transitions[INITIAL_STATE]:
                    self._init_config(curr_tag)
                
                # Update transitions from initial state 
                # if the tag is the first one in the sentence
                if i == 0:
                    self._transitions[INITIAL_STATE][curr_tag] += 1

                # Update the model if it is the first time to see next tag
                if sent[i+1][1] not in self._transitions[INITIAL_STATE]:
                    self._init_config(sent[i+1][1])

                # Update transitions for the current tag
                self._transitions[curr_tag][sent[i+1][1]] += 1

                # Update emissions for the current tag
                self._update_emissions(curr_tag, sent[i][0])
            
            # Update emissions fot the last tag in the sentence
            self._update_emissions(sent[-1][1], sent[-1][0])
        
        # Calculate transition and emission probabilities 
        # after done with processing all sentences
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
        """
        Method to update the model for the first encountered of a tag.
        ...

        Parameter:
        ----------
        tag : str
            Tag that is currently processing, might be current or next to the current.
        """
        # It is required to append new tag in the list of states.
        self._states.append(tag)

        # Add new entry for all already existing keys in transition dict,
        for key in self._transitions.keys():
            self._transitions[key][tag] = 0
        
        # Add key for new tag with entries for all already existing tags
        self._transitions[tag] = {state: 0 for state in self._states}
        
        # Add new key with all already seen words as entries if full_emissions
        # otherwise just create new key for new tag
        if self._full_emissions:
            self._emissions[tag] = {word: 0 for word in self._words}
        else:
            self._emissions[tag] = dict()

    def _update_emissions(self, tag, word):
        """
        Method to update emissions.
        ...

        Parameters:
        tag : str
            Current tag during processing
        word : str
            Word emissioned by current tag during processing
        """
        # If not already seen word is emissioned
        if word not in self._emissions[tag]:
            # If full_emissions it is required to add new word to all entries
            # not just to the current tag
            if self._full_emissions:
                self._words.append(word)
                for key in self._emissions.keys():
                    self._emissions[key][word] = 0
            else:
                self._emissions[tag][word] = 0

        self._emissions[tag][word] += 1

    def _calc_probs(self, sent_count):
        """
        Method to calculate probabilites after done with processing sentences.
        ...

        Parameter:
        sent_count : int
            Number of sentences in the train set
        """
        # In case of add_one smoothing it is required to know number of states and words
        if self._add_one:
            num_of_states = len(self._states)
            num_of_words = len(self._words)

        # Calculating probabilities for transitions
        for key in self._transitions.keys():
            # Different logic for transitions from the initial state because
            # normalization is done by number of sentences, while for other transitions
            # by number of states
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
        """
        Wrapper method around do_viterbi method to enable testing with more sentences.
        ...

        Parameters:
        test_path : str
            Directory where to find test set
        save_test_file : str
            File path to store output, words with corresponding predicted tags
        """
        corpus = ConllCorpusReader(test_path, ".t", ["words", "pos"])
        result = list()

        for sent in corpus.sents("de-test.t"):
            # Append end token if required
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
        """
        Method for performing viterbi algorithm on a sentence
        ...

        Parameters:
        -----------
        sentence : list
            List of words
        """
        # Need to clear trellis from the previous sentence during
        # whole test set processing
        self._trellis.clear_model()
        # Trellis starts with initial state
        self._trellis.add_timestep([self._initial_state])

        # Process word by word where one word is one timestep
        for timestep, word in enumerate(sentence):
            timestep_states = list()

            # Calculate for all possible states
            for state in self._states:
                backpointer = None
                curr_max_prob = -1

                for prev_state in self._trellis.get_timestep_states(timestep):
                    
                    # Unknown words crude handler
                    emission = 1.0
                    if word in self._emissions[state]:
                        emission = self._emissions[state][word]

                    # Calculate probability
                    prob = prev_state.get_max_prob() \
                        * self._transitions[prev_state.get_name()][state] \
                        * emission

                    # Check if it is the highest so far
                    # Remember backpointer
                    if prob > curr_max_prob:
                        curr_max_prob = prob
                        backpointer = prev_state.get_name()
                
                # Add a state to the current timestep
                timestep_states.append(State(state, backpointer=backpointer, max_prob=curr_max_prob))

            # Store the current timestep
            self._trellis.add_timestep(timestep_states)
        
        # Find the max probability for the whole trellis
        self._trellis.set_last_state()

        self._do_tagging()

    def _do_tagging(self):
        """
        Method to store the most probable tags for the current sentence.
        """
        self._tags.clear()
        curr_state = self._trellis.get_last_state()

        # Go backwards through trellis
        for timestep in list(reversed(self._trellis.get_model()[:-1])):
            self._tags.appendleft(curr_state.get_name())

            # Because a timestep is a list it is required to 
            # linearly check find state that correspondent to the backpointer 
            for state in timestep:
                if state.get_name() == curr_state.get_backpointer():
                    curr_state = state
                    break
    
    ############################
    # HELPER METHODS FOR CHECK #
    ############################

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

    def test_random_sents(self):
        corpus = ConllCorpusReader(self._data_path, ".tt", ["words", "pos"])

        for _ in range(4):
            index = np.random.randint(10, 500)
            sent = corpus.tagged_sents("de-train.tt")[index]

            self.do_viterbi([word[0] for word in sent])

            print(" ".join([word[0] for word in sent]))
            print(" ".join([word[1] for word in sent]))
            self.print_tags()
            print()

        
