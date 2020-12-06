Project file structure:
code/add_end.py
code/hmm.py
code/main.py
data/de-eval_end.tt
configs/base.json
configs/add_one.json
configs/add_one_end_token.json
configs/base_reduced.json
outputs/base.tt
outputs/add_one.tt
outputs/add_one_end_token.tt
outputs/base_reduced.tt
scores/base.txt
scores/add_one.txt
scores/add_one_end_token.txt
scores/base_reduced.txt
assignment_2_report.pdf
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
