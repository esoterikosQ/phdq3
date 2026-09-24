import json
import tempfile
import unittest
from pathlib import Path

from blt_hf.integrated_validation import gleu_improved, order_predictions, score_validation_epoch


class IntegratedValidationContracts(unittest.TestCase):
    def test_gleu_not_teacher_forced_loss_selects_best_epoch(self):
        self.assertTrue(gleu_improved(40., None))
        self.assertFalse(gleu_improved(39., 40.))
        self.assertFalse(gleu_improved(40., 40.))
        self.assertTrue(gleu_improved(41., 40.))

    def test_four_rank_predictions_score_full_corpus_in_original_order(self):
        sources = ['I has a nice red apple today .', 'This is a short sentence .',
                   'She walk to school each day .', 'Another sentence is correct .']
        references = ['I have a nice red apple today .', 'This is a short sentence .',
                      'She walks to school each day .', 'Another sentence is correct .']
        parts = [[{'index': i, 'text': references[i]}] for i in range(4)]
        self.assertEqual(order_predictions(parts, 4), references)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = score_validation_epoch(root, epoch=1, global_step=2,
                                            sources=sources, references=references,
                                            predictions=order_predictions(parts, 4),
                                            num_beams=1, max_new_bytes=768,
                                            validation_file_hash='fixture')
            self.assertEqual(result['gleu'], 100.)
            self.assertEqual(json.loads((root/'validation/epoch-0001/metrics.json').read_text())['gleu'], 100.)
            self.assertEqual((root/'validation/epoch-0001/hypothesis.txt').read_text().splitlines(), references)
            self.assertEqual(score_validation_epoch(root, epoch=1, global_step=2,
                                                   sources=sources, references=references,
                                                   predictions=references, num_beams=1,
                                                   max_new_bytes=768,
                                                   validation_file_hash='fixture'), result)

    def test_missing_duplicate_and_changed_predictions_are_rejected(self):
        with self.assertRaises(ValueError): order_predictions([[{'index': 0, 'text': 'a'}]], 2)
        with self.assertRaises(ValueError): order_predictions([[{'index': 0, 'text': 'a'}],
                                                               [{'index': 0, 'text': 'b'}]], 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = dict(epoch=1, global_step=2, sources=['I has apples .'],
                        references=['I have apples .'], num_beams=1,
                        max_new_bytes=768, validation_file_hash='fixture')
            score_validation_epoch(root, predictions=['I have apples .'], **args)
            with self.assertRaises(ValueError):
                score_validation_epoch(root, predictions=['I has apples .'], **args)
