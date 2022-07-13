import itertools
import operator
from collections import Counter
from math import log

from typing import List, Tuple, Dict

from interface.interface import MethodInterface


class TFIDF(MethodInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tf(self) -> List[Tuple[str, int]]:
        flat_list = list(itertools.chain(*self.segment_output))
        counter = Counter()
        for item in flat_list:
            if len(item) > 1:
                counter[item] += 1

        return sorted(counter.items(), key=operator.itemgetter(1), reverse=True)

    def idf(self) -> Tuple[List[float], Dict[str, int]]:
        tf = self.tf()
        tf_word = [w for w, v in tf]
        tf_value = [v for w, v in tf]

        df_list = []
        for word in tf_word:
            count = 0
            for sentence in self.segment_output:
                if word in sentence:
                    count += 1
            df_list.append(count)

        idf = []
        num_corpus = len(self.segment_output)
        for df in df_list:
            idf_value = log(num_corpus / df, 2)
            idf.append(idf_value)

        return idf, dict(zip(tf_word, tf_value))

    def tf_idf(self) -> List[Tuple[str, int]]:
        idf, tf = self.idf()
        tfidf_value = [i * j for i, j in zip(tf.values(), idf)]
        tfidf = dict(zip(tf.keys(), tfidf_value))
        tfidf_sorted = sorted(tfidf.items(), key=operator.itemgetter(1), reverse=True)
        return tfidf_sorted

    def run(self):
        return self.tf_idf()
