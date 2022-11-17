import numpy as np
import pickle
from fairseq.data import Dictionary

token = {2: '', 3: '' , 4: '', 5: '', 6 : '',
     7: '', 8:'', 9 : '', 10: '', 11: '', 12 : ''}
inv_map = {v: k for k, v in token.items()}


PMI = pickle.load(open("data/PMI_nltk.pkl","rb"))
grade_vocab = pickle.load(open("data/GradeVocab_nltk.pkl", "rb"))
tgt_dictionary = Dictionary.load("experiments/exp-1/data-bin/dict.tgt.txt")

words = tgt_dictionary.symbols

grades = range(2, 11)
grade_map = {grade:tgt_dictionary.index(token[grade]) for grade in grades}
inv_grade_map = {v: k for k, v in grade_map.items()}


weight_matrix = np.zeros((len(words), 12))

for i in range(len(words)):
    for grade in grades:
        if (words[i], grade) in PMI:
            weight_matrix[(i, grade)] = max(PMI[words[i], grade], 0) 
            
weight_matrix = weight_matrix+1

weight_matrix.dump("data/weight_matrix.npy")
pickle.dump(inv_grade_map, open("data/grade_map.pkl", "wb"))