import subprocess
import unittest
import os
from unittest.mock import patch
from pathlib import Path
from blt_hf.runtime import require_neuron_job, local_path
from blt_hf.train import parser as train_parser
from blt_hf.eval import parser as eval_parser

ROOT=Path(__file__).resolve().parents[1]
class NeuronContracts(unittest.TestCase):
    def test_off_cluster_execution_and_path_escape_fail(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):require_neuron_job()
        with self.assertRaises(ValueError):local_path('../outside')

    def test_clis_parse_without_gpu_import_or_network(self):
        self.assertEqual(train_parser().parse_args(['--run-dir','outputs/example']).epochs,3)
        self.assertTrue(eval_parser().parse_args(['--output-dir','outputs/example','--aggregate']).aggregate)

    def test_shell_syntax_and_scheduler_contract(self):
        for name in ('train_blt_hf.sh','eval_blt_hf.sh','score_blt_hf.sh','submit_blt_hf.sh','neuron_blt_hf_common.sh'):
            path=ROOT/'scripts'/name
            subprocess.run(['bash','-n',str(path)],check=True)
        for name in ('train_blt_hf.sh','eval_blt_hf.sh'):
            text=(ROOT/'scripts'/name).read_text()
            self.assertIn('#SBATCH --gres=gpu:1',text)
            self.assertIn('#SBATCH --signal=B:USR1@600',text)
        text=(ROOT/'scripts/submit_blt_hf.sh').read_text()
        self.assertIn('showque',text);self.assertIn('showappl',text)
        self.assertIn('--comment="field=$FIELD;appl=pytorch"',text)

    def test_config_fingerprint_ignores_only_loader_metadata(self):
        from blt_hf.runtime import model_config_identity
        class Config:
            def __init__(self,value):self.value=value
            def to_dict(self):return self.value
        a={'_name_or_path':'itcerdo/path','nested':{'window':512,'_name_or_path':'a'}}
        b={'_name_or_path':'neuron/path','nested':{'window':512,'_name_or_path':'b'}}
        self.assertEqual(model_config_identity(Config(a)),model_config_identity(Config(b)))
        b['nested']['window']=513
        self.assertNotEqual(model_config_identity(Config(a)),model_config_identity(Config(b)))
        factory=Config({'dtype':None,'nested':{'dtype':None}})
        loaded=Config({'dtype':'bfloat16','nested':{'dtype':'bfloat16'}})
        self.assertEqual(model_config_identity(factory,loader_dtype='bfloat16'),model_config_identity(loaded))
        self.assertNotEqual(model_config_identity(factory),model_config_identity(loaded))
        self.assertNotEqual(model_config_identity(Config({'dtype':'float32'}),loader_dtype='bfloat16'),
                            model_config_identity(Config({'dtype':'bfloat16'})))

    def test_conversion_evidence_is_bound_to_current_model_code(self):
        from blt_hf.runtime import conversion_verification, CONVERSION
        report=conversion_verification(ROOT/CONVERSION)
        self.assertEqual(report['weight_checks'],'passed')
        self.assertEqual(report['attention_mask_checks'],'passed')
