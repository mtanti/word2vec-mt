'''
Extract a vocabulary from the Maltese corpus.
'''

import collections
import tqdm
from word2vec_mt.constants import corpus_mt_path, vocab_mt_path


#########################################
MIN_FREQ_VOCAB = 5
'''
The minimum frequency a word must occur in the corpus to be included in the
vocabulary.
'''


#########################################
def extract_vocab(
) -> None:
    '''
    Extract a vocabulary consisting of all the tokens that occur at least
    MIN_FREQ_VOCAB times.
    '''
    print('Extracting vocabulary')
    try:
        with open(vocab_mt_path, 'x', encoding='utf-8') as f:
            pass
    except FileExistsError:
        print(f'Vocabulary file already exists: ({vocab_mt_path})')
        print('- Skipping')
        return

    print('- Counting lines in corpus')
    num_lines = 0
    with open(corpus_mt_path, 'r', encoding='utf-8') as f:
        for line in f:
            num_lines += 1

    print('- Reading corpus tokens')
    token_freqs = collections.Counter[str]()
    with open(corpus_mt_path, 'r', encoding='utf-8') as f:
        for (line, _) in zip(f, tqdm.tqdm(range(num_lines))):
            sent_tokens = line.strip().split(' ')
            token_freqs.update(sent_tokens)

    print('- Saving the vocabulary')
    with open(vocab_mt_path, 'w', encoding='utf-8') as f:
        for (token, freq) in sorted(
            token_freqs.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if freq >= MIN_FREQ_VOCAB:
                print(token, file=f)

    print('- Done')
