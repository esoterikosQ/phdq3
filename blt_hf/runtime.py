"""Project-local runtime guards, provenance, and transactional output helpers."""
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from .manifest import sha256_file, sha256_json, write_json

ROOT = Path(__file__).resolve().parents[1]
NEURON_ROOT = Path('/scratch/r984a02/phdq3')
CONVERSION = Path('blt_hf_checks/manifests/conversion_B_20260915.json')
MODEL = Path('artifacts/converted/blt-1b-hf-own')


def local_path(value, *, root=ROOT):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path must be inside project: {value}')
    return path


def require_neuron_job(*, gpu=True):
    if ROOT.resolve() != NEURON_ROOT.resolve() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run through SLURM in /scratch/r984a02/phdq3; no login-node training')
    partition = os.environ.get('SLURM_JOB_PARTITION', '')
    gpu_partitions = ('amd_a100nv_8', 'amd_a100_4', 'amd_h200nv_8')
    allowed = gpu_partitions if gpu else ('cpu', *gpu_partitions)
    if partition not in allowed:
        raise RuntimeError(f'Unreviewed partition: {partition}')
    if int(os.environ.get('SLURM_NNODES', '1')) != 1 or int(os.environ.get('SLURM_NTASKS', '1')) != 1:
        raise RuntimeError('This launcher supports one node and one SLURM task')
    if gpu:
        import torch
        expected_capability = (9, 0) if partition == 'amd_h200nv_8' else (8, 0)
        if not torch.cuda.is_available() or any(torch.cuda.get_device_capability(i) != expected_capability
                                              for i in range(torch.cuda.device_count())):
            raise RuntimeError(f'GPU capability does not match partition {partition}: expected {expected_capability}')
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f'Native BF16 is required on GPU {index}')


def code_identity():
    paths = sorted([*ROOT.glob('blt_hf/**/*.py'), *ROOT.glob('scripts/*blt_hf*.sh')])
    files = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = 'exported-source'
    # The digest prevents uncommitted source changes from sharing a fingerprint.
    return {'code_commit': commit + ':' + sha256_json(files), 'code_files': files}


def tokenizer_identity(model_path):
    names = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
    return sha256_json({name: sha256_file(model_path / name) for name in names if (model_path / name).exists()})


def atomic_json(path, value):
    """Replace mutable progress/pointers only; final evidence uses write_json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(value, out, ensure_ascii=False, allow_nan=False, indent=2)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def ensure_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing output has different identity: {path}')
    else:
        write_json(path, value)


@contextlib.contextmanager
def exclusive_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another job is writing {directory}') from exc
        try: yield
        finally: fcntl.flock(lock, fcntl.LOCK_UN)


class StopRequest:
    def __init__(self, max_seconds=0, reserve_seconds=600):
        self.requested = False
        self.deadline = time.monotonic() + max_seconds if max_seconds else float('inf')
        end = os.environ.get('SLURM_JOB_END_TIME')
        if end and end.isdigit():
            self.deadline = min(self.deadline, time.monotonic()+int(end)-time.time()-reserve_seconds)
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._handle)

    def _handle(self, *_): self.requested = True
    def __bool__(self): return self.requested or time.monotonic() >= self.deadline


def model_config_identity(config, *, loader_dtype=None):
    """Ignore loader location/provenance metadata, preserve every execution setting."""
    def clean(value):
        if isinstance(value, dict):
            return {k:(loader_dtype if k == 'dtype' and v is None and loader_dtype is not None else clean(v))
                    for k,v in value.items() if k not in ('_name_or_path', '_commit_hash')}
        if isinstance(value, list): return [clean(v) for v in value]
        return value
    return sha256_json(clean(config.to_dict()))


def conversion_verification(conversion):
    """Attach unchanged B/implementation to the completed conversion checks, scoped to itcerdo."""
    weights = ROOT/'blt_hf_checks/results/weights_p1_validation_20260915.json'
    masks = ROOT/'blt_hf_checks/results/p1_masks_20260916.json'
    w, m = json.loads(weights.read_text()), json.loads(masks.read_text())
    if w['status'] != 'passed' or w['conversion_report_hash'] != sha256_file(conversion):
        raise ValueError('Weight verification does not match the selected B artifact')
    if m['status'] != 'passed' or m['weight_evidence']['sha256'] != sha256_file(weights):
        raise ValueError('Mask verification is not linked to these weight checks')
    for name, expected in m['code_hashes'].items():
        if name.startswith('blt_hf/') and sha256_file(ROOT/name) != expected:
            raise ValueError(f'Runtime implementation changed since mask checks: {name}')
    return {'weight_checks':'passed', 'attention_mask_checks':'passed',
            'conversion_check_environment':'itcerdo/eager/bfloat16/no-cache/unpadded',
            'weight_report_hash':sha256_file(weights), 'mask_report_hash':sha256_file(masks),
            'original_forward_equivalence':'not_required_not_claimed'}
