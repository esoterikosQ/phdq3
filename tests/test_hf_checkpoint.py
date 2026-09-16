import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from blt_hf.checkpoint import resolve_checkpoint

class CheckpointContracts(unittest.TestCase):
    def test_pointer_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'latest.json';p.write_text('{"checkpoint":"../outside"}')
            with self.assertRaises(ValueError): resolve_checkpoint(p)

@unittest.skipUnless(importlib.util.find_spec('torch'), 'prepared HF environment required')
class CheckpointTensorTests(unittest.TestCase):
    def test_immutable_publication_strict_tensor_reload_and_tamper_detection(self):
        import torch
        from safetensors.torch import load_file
        from blt_hf.checkpoint import save_checkpoint
        class FakeOptimizer:
            def state_dict(self): return {'state':{},'param_groups':[]}
        with tempfile.TemporaryDirectory() as d:
            model=torch.nn.Linear(2,3)
            state={'global_step':1,'run_signature':'abc'}
            name=save_checkpoint(d,model,FakeOptimizer(),state,[{'cpu':torch.get_rng_state()}],best=True)
            path,meta=resolve_checkpoint(Path(d)/'best.json')
            self.assertEqual(path.name,name)
            restored=torch.nn.Linear(2,3)
            restored.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
            for a,b in zip(model.parameters(),restored.parameters()): self.assertTrue(torch.equal(a,b))
            self.assertEqual(torch.load(path/'training.pt',weights_only=True)['optimizer']['state'],{})
            (path/'model.safetensors').write_bytes(b'corrupted')
            with self.assertRaises(ValueError): resolve_checkpoint(Path(d)/'latest.json')
