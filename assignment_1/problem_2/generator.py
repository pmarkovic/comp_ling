import os
from nltk.probability import UniformProbDist


NO_CONTEXT = ()

class Generator:
    """
    """

    def __init__(self, model, texts_dir="assignment_1/problem_2/texts"):
        """
        """
        self._ngram_model = model
        self._texts_path = os.path.join(os.path.abspath(''), texts_dir)

        # For selecting starting words/context for a text
        if self._ngram_model.get_n() > 1:
            self._incipts = self._init_incipts()

    def _init_incipts(self):
        """
        Method which initialize the incipts (starting words/contexts) attribute.
        ...
        
        Returns:
        --------
        incipts_prob: UniformProbDist
            An instance of UniformProbDist class   
        """

        incipts = [context for context in self._ngram_model.contexts() if context[0].isupper()]
        incipts_prob = UniformProbDist(incipts)

        return incipts_prob


    def generate(self, num_of_texts, num_of_words=100, to_file=True, file_name="generated_text.txt"):
        """
        """
        
        while num_of_texts:
            if not self.generate_text(num_of_words, to_file):
                return False
        
            num_of_texts -= 1
        
        return True
    
    def generate_text(self, num_of_words=100, to_file=True, file_name="generated_text.txt"):
        """
        """
        
        # Check for unigrams
        if self._ngram_model.get_n() == 1:
            # Logic for unigram model
            text = [self._ngram_model[NO_CONTEXT].generate() for _ in range(num_of_words)]
        else:
            # Logic for n > 1 ngram models
            context = self._incipts.generate()
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
        try:
            with open(file_path, "w+") as text_file:
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
