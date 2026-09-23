"""Derive the Lang-8 prefix from union without changing the provided datasets."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from .data_adapter import dataset_split_path
from .manifest import sha256_file, sha256_json

LANG8_COUNTS = {'train': 76692, 'val': 16434, 'test': 16434}


def _rows(path):
    return [line.rstrip('\r\n') for line in Path(path).open(encoding='utf-8', newline=None)
            if line.strip()]


def _m2_records(path):
    records, current = [], []
    for raw in Path(path).open(encoding='utf-8'):
        line = raw.rstrip('\r\n')
        if not line:
            if current: records.append(current); current = []
            continue
        if line.startswith('S ') and current:
            records.append(current); current = []
        current.append(line)
    if current: records.append(current)
    if any(sum(line.startswith('S ') for line in record) != 1 for record in records):
        raise ValueError(f'Invalid logical M2 record in {path}')
    return records


def _publish(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ValueError(f'Existing derived artifact differs: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def derive(data_root, output_root=None):
    data_root = Path(data_root)
    output_root = Path(output_root) if output_root else data_root.parents[1]/'artifacts/derived/lang8'
    report = {'schema_version':1, 'derivation':'union-prefix-before-korean_learner-and-native', 'splits':{}}
    for split, expected in LANG8_COUNTS.items():
        union_path=data_root/'union'/f'union_{split}.txt'
        learner_path=data_root/'korean_learner'/f'korean_learner_{split}.txt'
        native_path=data_root/'native'/f'native_{split}.txt'
        union, learner, native = _rows(union_path), _rows(learner_path), _rows(native_path)
        if len(union)-len(learner)-len(native) != expected:
            raise ValueError(f'Unexpected Lang-8 count for {split}')
        if union[expected:] != learner + native:
            raise ValueError(f'Union suffix is not learner+native for {split}')
        output=output_root/f'lang8_{split}.txt'
        _publish(output, '\n'.join(union[:expected])+'\n')

        union_m2, learner_m2, native_m2 = (_m2_records(path.with_suffix('.m2'))
                                            for path in (union_path, learner_path, native_path))
        union_sources=[next(line[2:].strip() for line in record if line.startswith('S ')) for record in union_m2]
        suffix_sources=[next(line[2:].strip() for line in record if line.startswith('S '))
                        for record in learner_m2+native_m2]
        if (len(union_m2) != len(union) or union_sources[expected:] != suffix_sources
                or union_m2[expected:] != learner_m2 + native_m2):
            raise ValueError(f'Union M2 suffix is not learner+native for {split}')
        output_m2=output.with_suffix('.m2')
        _publish(output_m2, '\n\n'.join('\n'.join(record) for record in union_m2[:expected])+'\n')
        report['splits'][split]={'rows':expected,'tsv_sha256':sha256_file(output),
                                 'm2_sha256':sha256_file(output_m2),
                                 'source_hash':sha256_json(union_sources[:expected])}
    _publish(output_root/'derivation.json',json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/Preprocessed'))
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args(); report=derive(args.data_root,args.output_root)
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
