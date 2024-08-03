'''
A helper program to assist with manually constructing a similar word data
set consisting of a sample of (source) words and their similar words taken from
the vocabulary.
'''

import json
import random
import textwrap
from word2vec_mt.constants import (
    vocab_mt_path, vocab_en_path, synonyms_mt_path, translations_mten_path
)


#########################################

NUM_ENTRIES_NEEDED = 100
'''
The number of source words to include in the data set.
'''

NUM_SOURCE_WORD_SUGGESTIONS = 50
'''
The number of words to sample from the vocabulary to use as suggestions to the
user entering a source word.
'''


#########################################
def help_make_synonym_data_set(
) -> None:
    '''
    Help make a manually constructed synonymous words data set.
    '''
    print('Helping to construct a synonymous words data set')
    print()

    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = set(line.strip() for line in f)

    _help_make_similar_word_data_set(
        source_vocab=vocab_mt,
        similars_vocab=vocab_mt,
        output_path=synonyms_mt_path,
        seed=0,
    )


#########################################
def help_make_translations_data_set(
) -> None:
    '''
    Help make a manually constructed translated words data set.
    '''
    print('Helping to construct a translated words data set')
    print()

    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = set(line.strip() for line in f)
    with open(vocab_en_path, 'r', encoding='utf-8') as f:
        vocab_en = set(line.strip() for line in f)

    _help_make_similar_word_data_set(
        source_vocab=vocab_mt,
        similars_vocab=vocab_en,
        output_path=translations_mten_path,
        seed=1,
    )


#########################################
class Entry:
    '''
    Class for an entry (a source word and its similar words).
    '''

    #########################################
    def __init__(
        self,
        source: str,
        similars: list[str],
    ) -> None:
        '''
        Constructor.

        :param source: The source word.
        :param similars: The similar words.
        '''
        self.source = source
        self.similars = set(similars)

    #########################################
    def to_json(
        self,
    ) -> str:
        '''
        Convert this entry to a JSON line.

        :return: The JSON line.
        '''
        return json.dumps(
            {'source': self.source, 'similars': sorted(self.similars)},
            ensure_ascii=False,
        )


#########################################
def _load_file(
    source_vocab: set[str],
    similars_vocab: set[str],
    path: str,
) -> list[Entry]:
    '''
    Load a similar word data set file and validate it.

    :param source_vocab: The vocabulary to be used for the source words.
    :param similars_vocab: The vocabulary to be used for the similar words.
    :param path: The path to the jsonl file containing the data.
    :return: The loaded data.
    '''
    source: str
    similars: list[str]
    data_set: list[Entry] = []
    with open(path, 'r', encoding='utf-8') as f:
        for (line_num, line) in enumerate(f, 1):
            line = line.strip()
            try:
                doc = json.loads(line)
            except json.decoder.JSONDecodeError as ex:
                print(f'  Line {line_num}: Invalid JSON.')
                raise InvalidFileError() from ex
            if (
                not isinstance(doc, dict)
                or len(doc) != 2
                or 'source' not in doc
                or not isinstance(doc['source'], str)
                or 'similars' not in doc
                or not isinstance(doc['similars'], list)
                or not all(isinstance(x, str) for x in doc['similars'])
            ):
                print(
                    f'  Line {line_num}: JSON should be in the form of an'
                    ' object with source and similars as keys where source'
                    ' is associated with a string and similars is associated'
                    f' with a list of strings.'
                )
                raise InvalidFileError()
            source = doc['source']
            similars = doc['similars']

            if source not in source_vocab:
                print(
                    f'  Line {line_num}:'
                    f' The source word {source} is not in the vocabulary.'
                )
                raise InvalidFileError()
            for (line_num_, entry) in enumerate(data_set, 1):
                if source == entry.source or source in entry.similars:
                    print(
                        f'  Line {line_num}:'
                        f' The source word {source} was already used in entry'
                        f' {line_num_}.'
                    )
                    raise InvalidFileError()

            if len(similars) == 0:
                print(
                    f'  Line {line_num}:'
                    ' Must have at least one similar word.'
                )
                raise InvalidFileError()
            words: set[str] = {source}
            for similar in similars:
                if similar not in similars_vocab:
                    print(
                        f'  Line {line_num}:'
                        f' The similar word {similar} is not in the vocabulary.'
                    )
                    raise InvalidFileError()
                if similar in words:
                    print(
                        f'  Line {line_num}:'
                        f' The similar word {similar} is already in the entry.'
                    )
                    raise InvalidFileError()
            for i in range(1, len(similars)):
                if similars[i] < similars[i-1]:
                    print(
                        f'  Line {line_num}:'
                        f' The similar word {similars[i]} is not in sorted'
                        ' order.'
                    )
                    raise InvalidFileError()

            data_set.append(Entry(source, similars))

    return data_set


#########################################
def _help_make_similar_word_data_set(
    source_vocab: set[str],
    similars_vocab: set[str],
    output_path: str,
    seed: int,
) -> None:
    '''
    Help make a manually constructed similar word data set.
    A similar word data set consists of source words and at least one similar
    word for each source word.
    Words from the source vocabulary are randomly sampled to provide suggestions
    to the user when entering a source word.
    The program resumes from after last completed source word entry if
    terminated midway through.

    The source word and its similars are entered and validated by checking
    that:

    * The source word is in the source vocabulary.
    * The similar words are in the similars vocabulary.
    * There is at least one similar word.
    * The source word and its similars are all unique.
    * The source word is different from the words in previous entries.

    :param source_vocab: The vocabulary to be used for the source words.
    :param similars_vocab: The vocabulary to be used for the similar words.
    :param output_path: The path to the jsonl file containing the data.
    :param seed: The seed to use for the random number generator to randomly
        suggest words from the source vocabulary.
    '''
    previously_entered_data: list[Entry] = []
    shuffled_source_words = sorted(source_vocab)
    rng = random.Random(seed)
    rng.shuffle(shuffled_source_words)
    source: str
    similars: set[str]

    # Create the output file if it doesn't exist or load it if it does.
    try:
        with open(output_path, 'x', encoding='utf-8') as f:
            pass
    except FileExistsError:
        try:
            print('Loading existing data set')
            previously_entered_data = _load_file(
                source_vocab, similars_vocab, output_path,
            )
            print()
        except InvalidFileError:
            return

    # Ask user to enter new data.
    for i in range(1, NUM_ENTRIES_NEEDED + 1):
        # Skip already entered entries.
        if i <= len(previously_entered_data):
            continue

        suggested_source_words = shuffled_source_words[
            i*NUM_SOURCE_WORD_SUGGESTIONS:(i+1)*NUM_SOURCE_WORD_SUGGESTIONS
        ]
        print(
            'Working on source word'
            f' #{len(previously_entered_data)+1}/{NUM_ENTRIES_NEEDED}'
        )
        print('Suggested source words:')
        print(
            '    ' + '\n    '.join(
                textwrap.wrap(' '.join(suggested_source_words), 100)
            )
        )
        print()
        while True:
            source = _get_source_word(
                source_vocab, previously_entered_data,
            )
            similars = set()
            print()

            try:
                while True:
                    similar = _get_similar_word(
                        similars_vocab, {source}|similars,
                    )
                    similars.add(similar)
                    print()
            except EntryReady:
                entry = Entry(source, list(similars))
                with open(output_path, 'a', encoding='utf-8') as f:
                    print(entry.to_json(), file=f)
                    previously_entered_data.append(entry)
                print('  Saved.')
                print()
                break
            except CancelEntry:
                print('  Cancelled. Choose another source word.')
                print()

    print('  Done.')


#########################################
def _get_source_word(
    vocab: set[str],
    previously_entered_data: list[Entry],
) -> str:
    '''
    Ask the user for a source word.

    :param vocab: The vocabulary of possible words to use.
    :param previously_entered_data: The previously entered data.
    :return: The chosen source word.
    '''
    while True:
        source = input('  source: ')

        if source not in vocab:
            print('  This word is not in the vocabulary.')
        else:
            for (line_num, entry) in enumerate(previously_entered_data, 1):
                if source == entry.source or source in entry.similars:
                    print(
                        f'  This word was already used in entry {line_num}:'
                        f' {entry.source} - {" ".join(sorted(entry.similars))}'
                    )
                    break
            else:
                return source


#########################################
def _get_similar_word(
    vocab: set[str],
    words: set[str],
) -> str:
    '''
    Ask the user for a similar word of a source word.

    Will raise EntryReady if user enters the empty string and CancelEntry
    if user enters '.'.

    :param vocab: The vocabulary of possible words to use.
    :param words: The set of words consisting of the source word and its
        similar words entered up to now.
    :return: The chosen similar word.
    '''
    print(
        f'  Enter similar word #{len(words)} (or blank for done or . for'
        ' cancel).'
    )
    while True:
        similar = input('  similar: ')

        if similar == '':
            if len(words) == 1:
                print('  Must have at least one similar word.')
            else:
                raise EntryReady()
        elif similar == '.':
            raise CancelEntry()
        elif similar not in vocab:
            print('  This word is not in the vocabulary.')
        elif similar in words:
            print('  This word was already entered in the current entry.')
        else:
            return similar


#########################################
class InvalidFileError(Exception):
    '''
    Raised when a similar words data set file is invalid.
    '''


#########################################
class ControlSignal(Exception):
    '''
    Raised to indicate a user-generated signal for controlling the program.
    '''


#########################################
class EntryReady(ControlSignal):
    '''
    Raised when no more similars for a particular source word will be entered.
    '''


#########################################
class CancelEntry(ControlSignal):
    '''
    Raised when a particular source word should be cancelled and a new one used.
    '''
