#!/usr/bin/env python3
"""Generate the tokenizer differential corpus. Deterministic, offline."""
import itertools
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "corpus"
os.makedirs(OUT, exist_ok=True)

cases = {
    "ascii.txt": "The quick brown fox jumps over 13 lazy dogs. Don't stop, won't stop; it's o'clock.\n" * 200,
    "cyrillic.txt": "Привет мир! Это тест токенизатора. Ёжик и ёлка.\n" * 200,
    "cjk.txt": "你好世界。这是一个分词器测试。日本語のテスト。\n" * 200,
    "arabic.txt": "مرحبا بالعالم! هذا اختبار. ١٢٣٤٥\n" * 200,
    "emoji.txt": "a\U0001F600b\U0001F44D\U0001F3FDe\U0001F469\u200D\U0001F4BBf\n" * 200,
    "combining.txt": "e\u0301a\u0300o\u0308 n\u0303 u\u0308\n" * 200,
    "whitespace.txt": "a\t b\r\nc\r\nd  \n\ne   f\v\f g\n" * 200,
    "mixed.txt": "caf\u00e9 na\u00efve \u00d6l 8.5% \u2014 \"quoted\" [bracket] {brace} <tag>\n" * 200,
    "digits.txt": "1 22 333 4444 55555 666666 7777777 88888888\n" * 200,
    "punct.txt": "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n" * 200,
    "repeat.txt": ("word " * 50 + "\n") * 200,
    "longline.txt": ("x" * 5000 + "\n") * 20,
}

for name, text in cases.items():
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)
print(f"wrote {len(cases)} files to {OUT}")
