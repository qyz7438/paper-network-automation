import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CacheDefaultsTest(unittest.TestCase):
    def test_unset_and_empty_environment_use_default_cache_roots(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                env = os.environ.copy()
                for key in ('LLM_WIKI_MODELS_DIR', 'LLM_WIKI_TRANSLATION_MODEL_DIR'):
                    if empty:
                        env[key] = ''
                    else:
                        env.pop(key, None)
                code = ('import json; from scripts.paper_pipeline.local_models import DEFAULT_MODELS; '
                        'from scripts.paper_pipeline.query_translation import ROOT; '
                        'print(json.dumps([str(DEFAULT_MODELS), str(ROOT)], ensure_ascii=True))')
                run = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                                     capture_output=True, text=True, check=True)
                roots = json.loads(run.stdout)
                self.assertEqual(Path(roots[0]), Path.home() / '.cache' / 'llm-wiki' / 'models')
                self.assertEqual(Path(roots[1]), Path.home() / '.cache' / 'llm-wiki' / 'query-models')

    def test_model_roots_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env['LLM_WIKI_MODELS_DIR'] = str(Path(tmp) / 'models')
            env['LLM_WIKI_TRANSLATION_MODEL_DIR'] = str(Path(tmp) / 'translation')
            code = ('import json; from scripts.paper_pipeline.local_models import DEFAULT_MODELS; '
                    'from scripts.paper_pipeline.query_translation import ROOT; '
                    'print(json.dumps([str(DEFAULT_MODELS), str(ROOT)], ensure_ascii=True))')
            run = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                                 capture_output=True, text=True, check=True)
            roots = json.loads(run.stdout)
            self.assertEqual(Path(roots[0]), Path(env['LLM_WIKI_MODELS_DIR']))
            self.assertEqual(Path(roots[1]), Path(env['LLM_WIKI_TRANSLATION_MODEL_DIR']))


if __name__ == '__main__':
    unittest.main()
