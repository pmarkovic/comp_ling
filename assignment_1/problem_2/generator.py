import os
from nltk.probability import UniformProbDist


# Token for unigram model to be used as context.
NO_CONTEXT = ()

class Generator:
    """
    Class that is responsible for generating text by sampling from a ngram model.
    Generating is typically done by choosing incipt (starting context) first and then sampling from ngram model.
    For the unigram model, this step is omitted and sampling is done by using the NO_CONTEXT token for context.
    """

    def __init__(self, model, texts_dir="assignment_1/problem_2/texts"):
        """
        Class constructor.
        ...

        Parameters:
        -----------
        model : ngram
            Trained ngram model for sampling.
        texts_dir : str, optional
            Specify in which directory to find/create files for writing generated texts.
        """
        self._ngram_model = model
        # Path is: current working directory + path to directory
        self._texts_path = os.path.join(os.path.abspath(''), texts_dir)

        # For unigram models, incipts are not needed
        # For other ngram models, program needs to initialize incipts.
        if self._ngram_model.get_n() > 1:
            self._incipts = self._init_incipts()

    def _init_incipts(self):
        """
        Method which initialize the incipts (starting words/contexts) attribute.
        ...
        
        Returns:
        --------
        incipts_prob: UniformProbDist
            An instance of UniformProbDist class. Program choose the incipt from the uniform distribution. 
        """

        # Titles are all uppercase, so program will choose only such contexts to start the text.
        incipts = [context for context in self._ngram_model.contexts() if context[0].isupper()]
        incipts_prob = UniformProbDist(incipts)

        return incipts_prob


    def generate(self, num_of_texts, num_of_words=100, to_file=True, file_name="generated_text.txt"):
        """
        Wrapper around the generate_text method to allow generating more then one text in a program execution.
        ...

        Parameters:
        -----------
        num_of_texts : int
            Specify how many texts to be generated.
        num_of_words : int, optional
            Specify how many words per text to be generated.
        to_file : bool, optional
            Flag that indicates where to write generated texts. If False, text will be printed in terminal.
        file_name : str, optional
            Name of a file where generated texts will be written. 
            Note: it will be created/looked for in the above specified directory (texts_dir).
        """
        
        while num_of_texts:
            if not self.generate_text(num_of_words, to_file, file_name):
                return False
        
            num_of_texts -= 1
        
        return True
    
    def generate_text(self, num_of_words=100, to_file=True, file_name="generated_text.txt"):
        """
        Method responsible for generating a text.
        ...

        Parameters:
        -----------
        num_of_words : int, optional
            Specify how many words per text to be generated.
        file_name : str, optional
            Name of a file where generated texts will be written. 
            Note: it will be created/looked for in the above specified directory (texts_dir).
        """
        
        # Check for unigrams
        if self._ngram_model.get_n() == 1:
            # Logic for unigram model
            text = [self._ngram_model[NO_CONTEXT].generate() for _ in range(num_of_words)]
        else:
            # Logic for n > 1 ngram models
            context = self._incipts.generate()

            # All words are stored in a list to be later written as a text
            text = [word for word in context]
            num_of_words -= len(context)
            n = self._ngram_model.get_n()

            while num_of_words:
                next_word = self._ngram_model[context].generate()
                text.append(next_word)
                num_of_words -= 1

                context = tuple(text[-n+1:])
        
        # Add borders above and below the text
        hline = '-' * 50
        text = f"\n{hline}\n{' '.join(text)}\n{hline}\n"

        if not to_file:
            print(text)

            return True

        file_path = os.path.join(self._texts_path, file_name)
        print(file_path)
        try:
            with open(file_path, "a+") as text_file:
                text_file.write(text)
        except FileNotFoundError:
            print("File couldn't be open! Check file/directory path.")
    
            return False

        return True


    def get_incipts(self):
        """
        Getter method for the attribute incipts
        """
        return self._incipts
