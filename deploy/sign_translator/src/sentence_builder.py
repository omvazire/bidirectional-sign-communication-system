"""
Sentence Builder — NLP-based sentence formation from predicted gestures.

Accumulates predicted words and applies basic grammar rules to form
natural-sounding sentences from sign language gesture sequences.
"""

import re


# ─── Grammar Correction Rules ─────────────────────────────────
# Pattern -> Replacement mappings for common gesture combinations

GRAMMAR_RULES = [
    # ── Subject + state patterns ──
    (r"\bI sorry\b", "I am sorry"),
    (r"\bI good\b", "I am good"),
    (r"\bI bad\b", "I feel bad"),
    (r"\bYou good\b", "You are good"),
    (r"\bYou bad\b", "You are bad"),

    # ── Subject + want + action (most natural) ──
    (r"\bI want eat\b", "I want to eat"),
    (r"\bI want go home\b", "I want to go home"),
    (r"\bI want go\b", "I want to go"),
    (r"\bI want come\b", "I want to come"),
    (r"\bI want help\b", "I want help"),
    (r"\bI want water\b", "I want water"),
    (r"\bYou want eat\b", "Do you want to eat"),
    (r"\bYou want go\b", "Do you want to go"),
    (r"\bYou want water\b", "Do you want water"),
    (r"\bYou want come\b", "Do you want to come"),

    # ── Subject + action (no "want") ──
    (r"\bI eat\b", "I want to eat"),
    (r"\bI water\b", "I want water"),
    (r"\bI go home\b", "I am going home"),
    (r"\bI go\b", "I am going"),
    (r"\bI come\b", "I am coming"),
    (r"\bI love you\b", "I love you"),
    (r"\bI love home\b", "I love home"),
    (r"\bYou eat\b", "Do you want to eat"),
    (r"\bYou help\b", "Can you help"),
    (r"\bYou come\b", "Can you come"),
    (r"\bYou go home\b", "You can go home"),
    (r"\bYou go\b", "You can go"),

    # ── Command / Action requests ──
    (r"\bHelp I\b", "Help me"),
    (r"\bHelp you\b", "I can help you"),
    (r"\bPlease help\b", "Please help me"),
    (r"\bCome eat\b", "Come and eat"),
    (r"\bStop go\b", "Stop going"),

    # ── Descriptor + noun patterns ──
    (r"\bGood eat\b", "Good food"),
    (r"\bBad eat\b", "Bad food"),

    # ── Polite / Farewell patterns ──
    (r"\bThanks you\b", "Thank you"),
    (r"\bThanks help\b", "Thank you for helping"),
    (r"\bSorry no\b", "Sorry, no"),
    (r"\bGoodbye thanks\b", "Goodbye, thank you"),
    (r"\bGoodbye love\b", "Goodbye, love you"),
    (r"\bHello good\b", "Hello, how are you"),
]


class SentenceBuilder:
    """
    Builds grammatically reasonable sentences from gesture predictions.

    Features:
    - Word accumulation with duplicate filtering
    - Basic grammar correction using pattern matching
    - Capitalization rules (first word, "I")
    - Sentence finalization with punctuation
    """

    def __init__(self):
        self.words = []
        self.last_word = None
        self.sentence_history = []  # Previously spoken sentences

    def add_word(self, word):
        """
        Add a predicted word to the sentence.

        Filters consecutive duplicates to prevent stuttering.

        Args:
            word: Predicted gesture label (string)
        Returns:
            True if word was added, False if filtered
        """
        # Filter consecutive duplicates
        if word == self.last_word:
            return False

        self.words.append(word)
        self.last_word = word
        return True

    def get_raw_sentence(self):
        """Get the current sentence without grammar correction."""
        return " ".join(self.words)

    def get_corrected_sentence(self):
        """
        Get the current sentence with grammar corrections applied.

        Steps:
        1. Join words into raw sentence
        2. Apply pattern-based grammar rules
        3. Fix capitalization
        4. Add punctuation
        """
        if not self.words:
            return ""

        sentence = " ".join(self.words)

        # Apply grammar correction rules
        sentence = self._apply_grammar_rules(sentence)

        # Fix capitalization
        sentence = self._fix_capitalization(sentence)

        return sentence

    def get_final_sentence(self):
        """
        Get a finalized sentence ready for TTS output.
        Adds punctuation and stores in history.
        """
        sentence = self.get_corrected_sentence()
        if not sentence:
            return ""

        # Add period if no punctuation at end
        if sentence[-1] not in ".!?":
            # Certain patterns get question marks
            if sentence.lower().startswith(("do you", "can you", "are you")):
                sentence += "?"
            else:
                sentence += "."

        # Store in history
        self.sentence_history.append(sentence)

        return sentence

    def clear(self):
        """Clear the current sentence buffer."""
        self.words = []
        self.last_word = None

    def undo(self):
        """Remove the last added word."""
        if self.words:
            self.words.pop()
            self.last_word = self.words[-1] if self.words else None
            return True
        return False

    def get_word_count(self):
        """Get number of words in current sentence."""
        return len(self.words)

    def get_history(self):
        """Get list of previously spoken sentences."""
        return self.sentence_history.copy()

    # ─── Private Methods ──────────────────────────────────────

    def _apply_grammar_rules(self, sentence):
        """Apply pattern-based grammar correction rules."""
        for pattern, replacement in GRAMMAR_RULES:
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        return sentence

    def _fix_capitalization(self, sentence):
        """
        Fix capitalization:
        - Capitalize first letter
        - Always capitalize "I" as a standalone word
        - Capitalize after sentence-ending punctuation
        """
        if not sentence:
            return sentence

        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]

        # Always capitalize standalone "I"
        sentence = re.sub(r"\bi\b", "I", sentence)

        # Capitalize after period/exclamation/question
        sentence = re.sub(
            r"([.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            sentence,
        )

        return sentence


if __name__ == "__main__":
    # Demo
    builder = SentenceBuilder()

    # Simulate gesture predictions
    test_sequences = [
        ["Hello", "I", "Good"],
        ["I", "Want", "Eat"],
        ["I", "Want", "Go", "Home"],
        ["You", "Come"],
        ["I", "Love", "You"],
        ["Goodbye", "Thanks"],
        ["Please", "Help"],
        ["I", "Go", "Home"],
        ["Good", "Water"],
        ["Sorry", "No"],
    ]

    print("Sentence Builder Demo (20 Gestures):")
    print("-" * 50)

    for words in test_sequences:
        builder.clear()
        for w in words:
            builder.add_word(w)

        raw = builder.get_raw_sentence()
        corrected = builder.get_final_sentence()
        print(f"  Raw:       {raw}")
        print(f"  Corrected: {corrected}")
        print()
