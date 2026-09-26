"""The cache frontier must be diagnosed from observed patcher outputs."""
import unittest

from blt_hf_checks.check_patch_causality import compare_prefix, patch_starts


class PatchCausalityTests(unittest.TestCase):
    def test_next_token_slot_is_not_a_completed_patch(self):
        self.assertEqual(patch_starts([1, 3, 2], input_length=5), [0, 1, 4])
        self.assertEqual(patch_starts([1, 3, 2, 0], input_length=5), [0, 1, 4])
        with self.assertRaises(ValueError):
            patch_starts([1, 3, 1], input_length=5)

    def test_appending_byte_only_changes_open_patch_length(self):
        # Prefix length 4: [0, 1] with the final patch reaching the next-token slot.
        # Full length 6: the old final patch closes and a new one begins at 5.
        result = compare_prefix(
            full_entropies=[0.0, 0.2, 0.1, 0.3, 1.8, 0.1],
            full_lengths=[1, 4, 2],
            prefix_entropies=[0.0, 0.2, 0.1, 0.3],
            prefix_lengths=[1, 4],
        )
        self.assertTrue(result['starts_stable'])
        self.assertEqual(result['prefix_starts'], [0, 1])
        self.assertEqual(result['full_starts_through_prefix'], [0, 1])

    def test_detects_retroactive_boundary_change(self):
        result = compare_prefix(
            full_entropies=[0.0, 0.2, 1.8, 0.3, 0.1, 0.1],
            full_lengths=[1, 1, 5],
            prefix_entropies=[0.0, 0.2, 1.8, 0.3],
            prefix_lengths=[1, 1, 1, 2],
        )
        self.assertFalse(result['starts_stable'])
        self.assertEqual(result['prefix_starts'], [0, 1, 2, 3])
        self.assertEqual(result['full_starts_through_prefix'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
