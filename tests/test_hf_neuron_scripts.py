import subprocess
import unittest
import os
import tempfile
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
        self.assertNotIn('\nshowque\n',text);self.assertNotIn('\nshowappl\n',text)
        self.assertIn('BLT submit helper v2',text)
        self.assertIn('FIELD=${FIELD:-nlp}',text)
        self.assertIn('"--comment=field=${FIELD};appl=pytorch"',text)
        self.assertNotIn('--comment=\\"',text)

    def test_submit_reaches_sbatch_without_invoking_site_status_helpers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            bindir=root/'bin';bindir.mkdir()
            capture=root/'sbatch.args'
            commands={
                'hostname': '#!/bin/sh\nprintf "glogin01\\n"\n',
                'showque': '#!/bin/sh\nprintf "quota status\\n"\nexit 17\n',
                'showappl': '#!/bin/sh\nprintf "nlp\\n"\nexit 18\n',
                'squeue': '#!/bin/sh\nexit 0\n',
                'sbatch': '#!/bin/sh\nprintf "%s\\n" "$@" > "$SBATCH_CAPTURE"\n',
            }
            for name,body in commands.items():
                path=bindir/name;path.write_text(body);path.chmod(0o755)
            script=root/'submit.sh'
            source=(ROOT/'scripts/submit_blt_hf.sh').read_text()
            script.write_text(source.replace('cd /scratch/r984a02/phdq3',f'cd {root}'))
            env={**os.environ,'PATH':f'{bindir}:/usr/bin:/bin','USER':'r984a02',
                 'RUN_ID':'native-smoke-test','NUM_GPUS':'1','SBATCH_CAPTURE':str(capture)}
            result=subprocess.run(['bash',str(script),'smoke'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('BLT submit helper v2: mode=smoke field=nlp',result.stdout)
            self.assertNotIn('quota status',result.stdout)
            args=capture.read_text().splitlines()
            self.assertIn('--comment=field=nlp;appl=pytorch',args)
            self.assertNotIn('--comment="field=nlp;appl=pytorch"',args)
            self.assertIn('--partition=amd_a100nv_8',args)
            self.assertIn('--gres=gpu:1',args)
            self.assertEqual(args[-1],'scripts/train_blt_hf.sh')

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
