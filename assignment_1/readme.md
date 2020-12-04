Project file structur:
problem_1/zipf's_law.ipynb
problem_2/generator.py
problem_2/main.py
problem_2/problem_2_report.pdf
readme.md

Environment:
- python 3.8
- nltk 3.5
- manjaro 20.1

How to run code:
- For the problem 1 just open and run cells in notebook.
- For the problem 2 run the main.py script:
	python main.py n [-corpath str] [-textdir str] [-ntexts int] [-nwords int] [-tofile bool] [-filename str]
	 n        - N for ngram model (mandatory)
	-h        - for help
	-corpath  - Path to corpora directory (should specify, because default value is the path on my machine)
	-textdir  - Path to directory where to write generated text (if ommited, text will be written in the current working directory if -tofile True)
	-ntexts   - Number of texts to be generated
	-nwords   - Number of words per text to be generated
	-tofile   - Set True to write generated texts in a file, otherwise omit to write texts on stdout
	-filename - Name of a file to [be created to] write to generated texts

Runtimes:
- For 5 unigram texts     ~4.9sec
- For 10 bigram texts     ~5.4sec
- For 10 trigram texts    ~7.6sec
- For 20 quadrigram texts ~10.1sec
Note here: runtime is bounded by  model training. I also tried goodturing estimator, but it took more than 160sec for bigram model.
