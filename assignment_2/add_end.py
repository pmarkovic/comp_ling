from nltk.corpus.reader.conll import ConllCorpusReader

"""
Script to append end token to every sentence in eval set
in order to test properly.
New file is created.
"""

END_TOKEN     = "<END>"


corpus = ConllCorpusReader("data", ".tt", ["words", "pos"])
result = list()

for sent in corpus.tagged_sents("de-eval.tt"):
    sent.append((END_TOKEN, END_TOKEN))

    result.append(sent)

try:
    with open("./data/de-eval_end.tt", 'w') as conll_file:
        for sent in result:
            for pair in sent:
                conll_file.write("\t".join(pair)+'\n')
            conll_file.write('\n')
except FileNotFoundError:
    print("Not able to open the file for test writing!")
