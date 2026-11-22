"""Offline word autocomplete with local context learning.

No text is sent to a server. These completions are a small dictionary, not an LLM.
"""
import re
from collections import Counter
from pathlib import Path
import json


WORDS = ('HELLO', 'HELP', 'THANKS', 'PLEASE', 'WATER', 'KEYBOARD',
         'TURKEY', 'COMPUTER', 'PROJECT', 'FRIEND', 'GREAT', 'YES',
         'THANK', 'YOU', 'GOOD', 'MORNING', 'WOULD', 'LIKE', 'SOME',
         'NEED', 'NAME', 'TYPE', 'THIS', 'WITH', 'HAVE', 'THAT', 'WHAT',
         'WHERE', 'WHEN', 'THERE', 'THEIR', 'ABOUT', 'BECAUSE', 'PEOPLE',
         'SCHOOL', 'SCIENCE', 'TECHNOLOGY', 'HAPPY', 'TODAY', 'TOMORROW')

CORPUS = '''HELLO HOW ARE YOU
I AM HAPPY
I AM HUNGRY
I AM HERE
I HAVE A QUESTION
DO YOU HAVE A QUESTION
PLEASE HELP ME
I NEED HELP
I NEED WATER
I WOULD LIKE SOME WATER
THANK YOU FOR YOUR HELP
HAVE A GOOD DAY
GOOD MORNING EVERYONE
THIS IS MY TURKEY KEYBOARD
THE TURKEY CAN TYPE
I LIKE MY SCHOOL
MY SCIENCE PROJECT IS FUN
I AM WORKING ON MY PROJECT
I WANT TO TYPE
WHAT IS YOUR NAME
MY NAME IS
SEE YOU TOMORROW'''


class WordPredictor:
    """Small offline trigram model, with a local learned-word history.

    Only words followed by whitespace/punctuation are learned. Suggestions are
    never learned until committed; predictions never change an existing prefix.
    """
    def __init__(self, history_path=None):
        self.history_path = Path(history_path) if history_path else None
        self.history = []
        if self.history_path:
            try:
                data = json.loads(self.history_path.read_text(encoding='utf-8'))
                if isinstance(data, list):
                    self.history = [s[:4000] for s in data[-200:] if isinstance(s,str)]
            except (OSError, ValueError):
                pass
        here = Path(__file__).resolve().parent
        def read_bank(name):
            try:
                return '\n'.join(line for line in (here/name).read_text(encoding='utf-8').splitlines()
                                 if not line.lstrip().startswith('#'))
            except OSError:
                return ''
        self.base_words = WORDS + tuple(re.findall(r'[A-Z]+', read_bank('word_bank.txt').upper()))
        self.phrases = CORPUS.splitlines()+read_bank('context_phrases.txt').splitlines()
        self._rebuild()

    def _rebuild(self):
        self.unigrams = Counter(self.base_words)
        self.bigrams = Counter()
        self.trigrams = Counter()
        for sample in self.phrases+self.history:
            # Sentence endings are boundaries; do not learn across them.
            for sentence in re.split(r'[.!?\n]', sample.upper()):
                words = re.findall(r'[A-Z]+', sentence)
                self.unigrams.update(words)
                self.bigrams.update(zip(words, words[1:]))
                self.trigrams.update(zip(words, words[1:], words[2:]))
        self.vocabulary_size = len(self.unigrams)

    def learn(self, text):
        # Save only committed context, locally. Limit the history size.
        if not text or text[-1].isalpha():
            return
        sample = text[-4000:]
        if self.history and sample == self.history[-1]:
            return
        # Replace an earlier snapshot of the same sentence rather than counting
        # its first words again at every space.
        if self.history and sample.startswith(self.history[-1]):
            self.history[-1] = sample
        else:
            self.history.append(sample)
        self.history = self.history[-200:]
        self._rebuild()
        if self.history_path:
            try:
                self.history_path.write_text(json.dumps(self.history), encoding='utf-8')
            except OSError:
                pass

    def suggest(self, text):
        line = text.rsplit('\n',1)[-1].upper()
        if not line.strip():
            return ''
        match = re.search(r'[A-Z]+$', line)
        prefix = match.group() if match else ''
        context = re.findall(r'[A-Z]+', line[:match.start()] if match else line)
        if len(prefix) == 1:
            return ''
        unigrams, bigrams, trigrams = self.unigrams, self.bigrams, self.trigrams
        def rank(word):
            tri = trigrams[tuple(context[-2:])+(word,)] if len(context)>=2 else 0
            bi = bigrams[(context[-1],word)] if context else 0
            return tri, bi, unigrams[word]
        candidates = [word for word in unigrams if word.startswith(prefix) and word != prefix]
        if not prefix:
            candidates = [word for word in candidates if rank(word)[:2] != (0,0)]
        if not candidates:
            return ''
        best = max(sorted(candidates), key=rank)
        return best[len(prefix):]
