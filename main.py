from builder import Builder

if __name__ == '__main__':
    # TFIDF
    result = Builder.run('tfidf')
    print(result)

    # N-gram
    w = Builder.run('ngram', n=2)
    text = '女'
    next_words = list(w[text])[0:5]
    for next_word in next_words:
        print(f"next word: {next_word.word}, probability: {next_word.prob}")
    # result = w.ngram()






