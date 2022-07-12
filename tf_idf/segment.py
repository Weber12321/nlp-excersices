from typing import Iterable, List

import jieba as jb


def word_segment(data: Iterable[str]) -> List[str]:
    return [jb.cut(d, cut_all=True) for d in data]
