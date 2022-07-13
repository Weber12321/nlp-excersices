from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Union, Iterable

import jieba as jb

import pandas as pd


class MethodInterface(ABC):
    def __init__(self, data_path: Union[str, Path] = None, **kwargs):
        self.data_path = data_path
        self.data = self.read_file()
        self.stop_word: Iterable[str] = kwargs.get('stop_word', None)
        self.segment_output: List[List[str]] = self.segment()

    def read_file(self) -> List[str]:
        data_path = self.data_path if self.data_path else 'data/data.json'
        df = pd.read_json(data_path)
        return df.text.values.tolist()

    def segment(self) -> List[List[str]]:
        if self.stop_word:
            corpus = []
            for d in self.data:
                seg = jb.cut(d, cut_all=True)
                segment = list(filter(lambda x: x not in self.stop_word, seg))
                corpus.append(segment)
            return corpus
        else:
            return [list(jb.cut(d, cut_all=True)) for d in self.data]

    @abstractmethod
    def run(self) -> Any:
        """run the calculation"""
        pass
