#!/usr/bin/env python3
"""Extract Chinese text from WPS .doc binary files (UTF-16LE heuristic scan)."""
import re
import sys
from pathlib import Path

def extract_utf16_chinese(data: bytes) -> str:
    """Scan bytes for UTF-16LE runs of CJK/ASCII printable text."""
    text_parts = []
    i = 0
    n = len(data)
    run = []
    while i + 1 < n:
        code = data[i] | (data[i + 1] << 8)
        ch = None
        if 0x4E00 <= code <= 0x9FFF:      # CJK Unified
            ch = chr(code)
        elif 0x3000 <= code <= 0x303F:    # CJK punctuation
            ch = chr(code)
        elif 0xFF00 <= code <= 0xFFEF:    # Fullwidth forms
            ch = chr(code)
        elif code in (0x2018, 0x2019, 0x201C, 0x201D, 0x2014, 0x2026, 0x00B7):
            ch = chr(code)
        elif 0x20 <= code <= 0x7E:        # ASCII printable
            ch = chr(code)
        elif code in (0x000D, 0x000A, 0x0009):
            ch = '\n' if code == 0x000D else ch
        else:
            ch = None
        if ch is not None:
            run.append(ch)
            i += 2
        else:
            if len(run) >= 6:  # keep runs of decent length
                text_parts.append(''.join(run))
            run = []
            i += 2
    if len(run) >= 6:
        text_parts.append(''.join(run))
    return '\n'.join(text_parts)

def extract_gbk_chinese(data: bytes) -> str:
    """Fallback: scan for GBK-encoded CJK runs."""
    text_parts = []
    i = 0
    n = len(data)
    run = []
    while i < n:
        b = data[i]
        if 0x81 <= b <= 0xFE and i + 1 < n and 0x40 <= data[i + 1] <= 0xFE:
            try:
                ch = bytes([b, data[i + 1]]).decode('gbk')
                run.append(ch)
                i += 2
                continue
            except UnicodeDecodeError:
                pass
        if len(run) >= 6:
            text_parts.append(''.join(run))
        run = []
        i += 1
    if len(run) >= 6:
        text_parts.append(''.join(run))
    return '\n'.join(text_parts)

def clean(text: str) -> str:
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # keep lines containing at least 2 CJK chars
        cjk = len(re.findall(r'[\u4e00-\u9fff]', line))
        if cjk >= 2:
            lines.append(line)
    return '\n'.join(lines)

def main():
    out_dir = Path('/workspace/extracted')
    out_dir.mkdir(exist_ok=True)
    files = sorted(Path('/workspace').glob('*.md'),
                   key=lambda p: int(re.match(r'(\d+)', p.name).group(1)))
    for f in files:
        data = f.read_bytes()
        t16 = extract_utf16_chinese(data)
        t16 = clean(t16)
        if len(re.findall(r'[\u4e00-\u9fff]', t16)) < 10:
            tg = extract_gbk_chinese(data)
            t16 = clean(tg)
        out = out_dir / f.name
        out.write_text(t16, encoding='utf-8')
        cjk = len(re.findall(r'[\u4e00-\u9fff]', t16))
        print(f'{f.name}: {cjk} CJK chars extracted')

if __name__ == '__main__':
    main()
