train.tsv/dev.tsv/test.tsv are our split of the original "Quora Sentence Pairs" dataset (https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs).

Each line of these files represents a question pair, and includes four tab-seperated fields: 
judgement, question_1_toks, question_2_toks, pair_ID(from the orignial file)

Inside these files, all questions are tokenized with Stanford CoreNLP toolkit.

"wordvec.txt" is a tiny word vector model customized for our "train/dev/test" sets, and the word vectors are extracted from a huge model at http://nlp.stanford.edu/data/glove.42B.300d.zip

Please cite our paper "Bilateral Multi-Perspective Matching for Natural Language Sentences" when you use this data split.

