# custom script for kokoro removing phonemizer dependency, which had excessive sub-dependencies; full explanation at the bottom

import re
import torch
import numpy as np
import subprocess
import os

def split_num(num):
    num = num.group()
    if '.' in num:
        return num
    elif ':' in num:
        h, m = [int(n) for n in num.split(':')]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f'{h} oh {m}'
        return f'{h} {m}'
    year = int(num[:4])
    if year < 1100 or year % 1000 < 10:
        return num
    left, right = num[:2], int(num[2:4])
    s = 's' if num.endswith('s') else ''
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f'{left} hundred{s}'
        elif right < 10:
            return f'{left} oh {right}{s}'
    return f'{left} {right}{s}'

def flip_money(m):
    m = m.group()
    bill = 'dollar' if m[0] == '$' else 'pound'
    if m[-1].isalpha():
        return f'{m[1:]} {bill}s'
    elif '.' not in m:
        s = '' if m[1:] == '1' else 's'
        return f'{m[1:]} {bill}{s}'
    b, c = m[1:].split('.')
    s = '' if b == '1' else 's'
    c = int(c.ljust(2, '0'))
    coins = f"cent{'' if c == 1 else 's'}" if m[0] == '$' else ('penny' if c == 1 else 'pence')
    return f'{b} {bill}{s} and {c} {coins}'

def point_num(num):
    a, b = num.group().split('.')
    return ' point '.join([a, ' '.join(b)])

def expand_acronym(m):
    text = m.group(0).replace('.', '')
    letters = list(text)
    letters_with_periods = [letter + '.' for letter in letters]
    return ' '.join(letters_with_periods)

def normalize_text(text):
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace('«', chr(8220)).replace('»', chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = text.replace('(', '«').replace(')', '»')
    for a, b in zip('、。！，：；？', ',.!,:;?'):
        text = text.replace(a, b+' ')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    text = re.sub(r'\betc\.(?! [A-Z])', 'etc', text)
    text = re.sub(r'(?i)\b(y)eah?\b', r"\1e'a", text)
    text = re.sub(r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)', split_num, text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b', flip_money, text)
    text = re.sub(r'\d*\.\d+', point_num, text)
    text = re.sub(r'(?<=\d)-(?=\d)', ' to ', text)
    text = re.sub(r'(?<=\d)S', ' S', text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", 's', text)
    text = re.sub(r'(?:[A-Za-z]\.){2,} [a-z]', lambda m: m.group().replace('.', '-'), text)
    text = re.sub(r'(?i)(?<=[A-Z])\.(?=[A-Z])', '-', text)
    text = re.sub(r'\b(?:[A-Z]\.?){2,}\b', expand_acronym, text)
    return text.strip()

def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts

VOCAB = get_vocab()

def tokenize(ps):
    return [i for i in map(VOCAB.get, ps) if i is not None]

def direct_espeak(text, lang='en-us'):
    espeak_path = r"D:\Scripts\bench_tts\hexgrad--Kokoro-82M\espeak-ng.exe" # NEED TO MAKE DYNAMIC OR RELATIVE USING PATHLIB
    cmd = [espeak_path, '-q', '--ipa', '-v', lang, text]
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            shell=True
        )
        if result.returncode != 0:
            print(f"Espeak error: {result.stderr}")
            return ''
        return result.stdout.strip()
    except Exception as e:
        print(f"Error running espeak: {e}")
        return ''

def phonemize(text, lang, norm=True):
    if norm:
        text = normalize_text(text)
    
    espeak_lang = 'en-us' if lang == 'a' else 'en-gb'
    ps = direct_espeak(text, espeak_lang)
    if not ps:
        return ''
        
    ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
    ps = re.sub(r'(?<=[a-zɹː])(?=hˈʌndɹɪd)', ' ', ps)
    ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', 'z', ps)
    if lang == 'a':
        ps = re.sub(r'(?<=nˈaɪn)ti(?!ː)', 'di', ps)
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    return ps.strip()

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

@torch.no_grad()
def forward(model, tokens, ref_s, speed):
    device = ref_s.device
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
        c_frame += pred_dur[0,i].item()
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
    return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()

def generate(model, text, voicepack, lang='a', speed=1, ps=None):
    ps = ps or phonemize(text, lang)
    tokens = tokenize(ps)
    if not tokens:
        return None
    elif len(tokens) > 510:
        tokens = tokens[:510]
        print('Truncated to 510 tokens')
    ref_s = voicepack[len(tokens)]
    out = forward(model, tokens, ref_s, speed)
    ps = ''.join(next(k for k, v in VOCAB.items() if i == v) for i in tokens)
    return out, ps

def generate_full(model, text, voicepack, lang='a', speed=1, ps=None):
    ps = ps or phonemize(text, lang)
    tokens = tokenize(ps)
    if not tokens:
        return None
    outs = []
    loop_count = len(tokens)//510 + (1 if len(tokens) % 510 != 0 else 0)
    for i in range(loop_count):
        ref_s = voicepack[len(tokens[i*510:(i+1)*510])]
        out = forward(model, tokens[i*510:(i+1)*510], ref_s, speed)
        outs.append(out)
    outs = np.concatenate(outs)
    ps = ''.join(next(k for k, v in VOCAB.items() if i == v) for i in tokens)
    return outs, ps

"""
THE ORIGINAL SCRIPT relied on "phonemizer" a wrapper around espeak-ng

```python
phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
    b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True),
)
```
- language selection (en-us or en-gb)
- preserve_punctuation=True (keep punctuation in output)
- with_stress=True (keep stress markers in phonemes)

This modified script uses espeak-ng directly, providing the appropriate flags for identical functionality:
```python
def direct_espeak(text, lang='en-us'):
    espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    cmd = [espeak_path, '-q', '--ipa', '-v', lang, text]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', shell=True)
```

- `-q`: Quiet mode (no audio output)
- `--ipa`: Output in International Phonetic Alphabet notation
- `-v lang`: Language selection (en-us or en-gb)

THE ORIGINAL SCRIPT set certain variables:
```python
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
```

- PHONEMIZER_ESPEAK_LIBRARY pointed to the DLL file that phonemizer used to interface with espeak
- PHONEMIZER_ESPEAK_PATH pointed to the espeak executable that phonemizer would call

This modified script does not because we are no longer using the phonemize library, which requires
1. This script specifically sets the path to the espeak binary.
2. When distribution it'll be necessary to either bundle the binary and copy it to the same directory in which kokoro.py is located
(in which case amending the PATH is not necessary) or add functionality to search for and locate espeak.

espeak-ng requires three files to function:
espeak-ng-data -- all the languages and stuff
espeak-ng.exe - binary that links to the .dll
libespeak-ng.dll
"""
