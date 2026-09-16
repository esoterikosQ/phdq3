import unittest
from blt_hf.generation import GenerationConfig, decode_generated, group_prompts

class GenerationContracts(unittest.TestCase):
    def test_exact_length_groups_preserve_original_indices(self):
        prompts = [[1, 8], [1, 9, 10], [1, 4], [1, 9, 7]]
        groups = group_prompts(prompts, 2)
        self.assertEqual(sorted(i for group in groups for i in group), list(range(4)))
        self.assertTrue(all(len({len(prompts[i]) for i in group}) == 1 for group in groups))

    def test_eos_empty_and_utf8_budget_are_reported(self):
        self.assertEqual(decode_generated([2, 3, 3])["text"], "")
        self.assertTrue(decode_generated([2])["eos_reached"])
        ids = [b+4 for b in '가'.encode()]
        self.assertEqual(decode_generated(ids+[2])["text"], '가')
        bad = decode_generated(ids[:2])
        self.assertFalse(bad['valid_utf8'])
        self.assertFalse(bad['eos_reached'])
        self.assertIn('\ufffd', bad['text'])
        self.assertEqual(decode_generated([4+10, 4+65, 4+13, 2])['text'], 'A')
        self.assertFalse(decode_generated([1, 10, 2])['valid_token_ids'])

    def test_cache_and_padding_are_not_silently_enabled(self):
        with self.assertRaises(ValueError): GenerationConfig(use_cache=True)
        with self.assertRaises(ValueError): GenerationConfig(num_beams=0)

import importlib.util
@unittest.skipUnless(importlib.util.find_spec('torch'), 'prepared HF environment required')
class ActualGenerationTests(unittest.TestCase):
    def test_tiny_hf_greedy_and_beam_batch_order_and_invariance(self):
        import torch
        from test_hf_model import ModelTests
        from test_hf_data_adapter import ByteTokenizer
        from blt_hf.patched.modeling_blt import BltForCausalLM
        from blt_hf.generation import generate_batch
        torch.manual_seed(19)
        model=BltForCausalLM(ModelTests().config('osc')).eval()
        sources=['a','abcd','b']
        for beams in (1,4):
            cfg=GenerationConfig(batch_size=3,num_beams=beams,max_new_bytes=4)
            together=generate_batch(model,ByteTokenizer(),sources,cfg)
            single=[generate_batch(model,ByteTokenizer(),[source],GenerationConfig(num_beams=beams,max_new_bytes=4))[0] for source in sources]
            self.assertEqual(together,single)
