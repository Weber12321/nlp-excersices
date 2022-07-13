from pathlib import Path
from typing import Union, List

import pandas as pd

from n_gram.n_gram import Ngram
from tf_idf.tf_idf import TFIDF


class Builder:
    @staticmethod
    def run(method: str, **kwargs):
        if method == 'tfidf':
            worker = TFIDF()
            return worker.run()
        elif method == 'ngram':
            worker = Ngram(kwargs.get('n', None))
            return worker.run()
        else:
            raise ValueError(f"method {method} not supported")

