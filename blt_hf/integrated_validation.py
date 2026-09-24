"""Complete-corpus GLEU selection inside a distributed training job."""
from pathlib import Path

from .evaluation import publish_lines
from .manifest import sha256_file
from .metrics import compute_gleu
from .runtime import ensure_json


def gleu_improved(score, previous_best):
    """Keep the earlier epoch on an exact GLEU tie."""
    if score is None:
        return False
    return previous_best is None or score > previous_best


def order_predictions(rank_parts, total):
    """Reassemble every validation row exactly once, independent of rank order."""
    if total < 1:
        raise ValueError('Validation set must be nonempty')
    ordered = [None] * total
    for part in rank_parts:
        for item in part:
            index, value = item['index'], item['text']
            if not isinstance(index, int) or not 0 <= index < total or ordered[index] is not None:
                raise ValueError(f'Duplicate or out-of-range validation prediction: {index}')
            if not isinstance(value, str):
                raise ValueError(f'Non-text validation prediction: {index}')
            ordered[index] = value
    if any(value is None for value in ordered):
        raise ValueError('Missing validation predictions')
    return ordered


def score_validation_epoch(run_dir, *, epoch, global_step, sources, references,
                           predictions, num_beams, max_new_bytes, validation_file_hash):
    if not sources or len({len(sources), len(references), len(predictions)}) != 1:
        raise ValueError('Validation GLEU requires the complete aligned split')
    root = Path(run_dir) / 'validation' / f'epoch-{epoch:04d}'
    source = root / 'source.txt'
    reference = root / 'reference.txt'
    hypothesis = root / 'hypothesis.txt'
    publish_lines(source, sources)
    publish_lines(reference, references)
    publish_lines(hypothesis, predictions)
    result = {'epoch': epoch, 'global_step': global_step, 'split': 'val',
              'sample_count': len(sources), 'num_beams': num_beams,
              'max_new_bytes': max_new_bytes, 'validation_file_hash': validation_file_hash,
              'source_hash': sha256_file(source), 'reference_hash': sha256_file(reference),
              'hypothesis_hash': sha256_file(hypothesis),
              'gleu': compute_gleu(reference, source, hypothesis)}
    ensure_json(root / 'metrics.json', result)
    return result
