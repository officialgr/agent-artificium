from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from artificium.cli import main, _stop_background
from artificium.config import Config, ConfigStore
from artificium.engine import make_engine, EngineError
from artificium.filesystem import Paths, atomic_write_json
from artificium.initialization import initialize_mind
from artificium.operator import status_snapshot
from artificium.prompts import PromptPack
from artificium.records import Records
from artificium.setup import SetupWizard, SetupOptions, ModelDiscovery, resolved_vision

ROOT = Path(__file__).resolve().parents[2]


class Response:
    def __init__(self, value):
        self.value = value
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return json.dumps(self.value).encode()


class ReleaseCase(unittest.TestCase):
    def setUp(self):
        # These unit cases isolate metadata/core behavior. Real connection
        # verification and failures are covered over HTTP in test_connection_flow.
        check = mock.patch('artificium.setup.verify_connection', side_effect=lambda paths, config, key, **kw: (config, {}))
        check.start()
        self.addCleanup(check.stop)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(self.temp.cleanup)
        self.paths = Paths(Path(self.temp.name) / 'instance')
        shutil.copytree(ROOT / 'artificium-code/prompts', self.paths.prompts)
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        network = mock.patch('urllib.request.urlopen', side_effect=OSError('offline test'))
        network.start()
        self.addCleanup(network.stop)

    def llama_server(self, vision=False, context=32768, owner='llamacpp'):
        self.requests = []
        def respond(request, timeout=0):
            self.requests.append(request)
            if request.full_url.endswith('/models'):
                return Response({'data':[{'id':'local-model','owned_by':owner,
                                         'meta':{'n_ctx_train':131072}}]})
            if '/props?' in request.full_url:
                return Response({'default_generation_settings':{'n_ctx':context},
                                 'modalities':{'vision':vision}})
            raise AssertionError(request.full_url)
        return mock.patch('urllib.request.urlopen', side_effect=respond)

    def test_llamacpp_discovery_uses_serving_capacity_and_text_only_metadata(self):
        with self.llama_server():
            result = SetupWizard(self.paths).run(SetupOptions(provider='llamacpp'))
        config = ConfigStore(self.paths).load()
        self.assertEqual((config.model, config.context_window_tokens, config.vision), ('local-model',32768,'no'))
        self.assertEqual(config.adapter, 'llamacpp')
        self.assertEqual(len(self.requests), 2)
        self.assertFalse(self.paths.secrets.exists())
        self.assertEqual(result['provider'], 'llamacpp')

    def test_custom_llama_server_gets_native_mapping_and_vision_discovery(self):
        with self.llama_server(vision=True):
            SetupWizard(self.paths).run(SetupOptions(provider='custom',api_url='http://localhost:8080',reasoning_effort='low'))
        config = ConfigStore(self.paths).load()
        self.assertEqual((config.provider, config.adapter, config.vision), ('custom','llamacpp','auto'))
        self.assertEqual(make_engine(config,None).prepare([]).payload['reasoning_effort'],'low')

    def test_llamacpp_no_props_uses_an_explicitly_unverified_default_budget(self):
        discovery = ModelDiscovery(models=('model',), details={'model':{'owned_by':'llamacpp'}})
        with mock.patch('artificium.setup.discover_provider_models',return_value=discovery):
            config, _ = SetupWizard(self.paths)._build(SetupOptions(provider='llamacpp'))
        self.assertEqual((config.context_window_tokens, config.context_window_source), (32768, 'default'))
        self.assertFalse(self.paths.config.exists())

    def test_context_above_actual_server_limit_is_rejected_before_save(self):
        with self.llama_server(), self.assertRaisesRegex(ValueError,'exceeds'):
            SetupWizard(self.paths).run(SetupOptions(provider='llamacpp',context_window_tokens=65536))
        self.assertFalse(self.paths.config.exists())

    def test_manual_vision_override_is_preserved(self):
        for requested in ('yes','no'):
            with self.llama_server(vision=requested=='no'):
                config,_ = SetupWizard(self.paths)._build(SetupOptions(provider='llamacpp',vision=requested))
            self.assertEqual(config.vision_preference,requested)
            self.assertEqual(config.vision,'no')
        self.assertEqual(resolved_vision('custom','auto',{}),'no')
        self.assertEqual(resolved_vision('openrouter','auto',{'vision':False}),'no')
        self.assertEqual(resolved_vision('openrouter','auto',{}),'auto')

    def test_setup_and_configure_share_editor_without_repeated_discovery(self):
        with self.llama_server(), mock.patch('builtins.input',return_value=''), redirect_stdout(io.StringIO()):
            SetupWizard(self.paths).run(SetupOptions(provider='llamacpp'),interactive=True)
        self.assertEqual(len(self.requests),2)
        before = self.paths.self_file.read_bytes()
        # Current provider's menu choice, current URL, advanced=no, save=yes.
        with self.llama_server(), mock.patch('builtins.input',return_value=''), redirect_stdout(io.StringIO()):
            SetupWizard(self.paths).reconfigure(SetupOptions(),interactive=True)
        self.assertEqual(len(self.requests),2)
        self.assertEqual(self.paths.self_file.read_bytes(),before)

    def test_setup_does_not_replace_existing_mind_or_memory(self):
        self.paths.self_file.parent.mkdir(parents=True)
        self.paths.self_file.write_text('My existing Self.')
        self.paths.meta_memory.write_text('My learned memory map.')
        memory = self.paths.memory/'harness/infinite-attention.txt'
        memory.parent.mkdir(parents=True)
        memory.write_text('A verified learned strategy.')
        seed = self.paths.prompts/'mind-seed/memory/harness/infinite-attention.txt'
        seed.write_text('Updated starting advice for new instances.')
        with self.llama_server():
            SetupWizard(self.paths).run(SetupOptions(provider='llamacpp'))
        self.assertEqual(self.paths.self_file.read_text(),'My existing Self.')
        self.assertEqual(self.paths.meta_memory.read_text(),'My learned memory map.')
        self.assertEqual(memory.read_text(),'A verified learned strategy.')

    def test_switching_provider_can_select_its_single_model(self):
        ConfigStore(self.paths).save(Config(provider='custom', model='old', base_url='http://localhost:8000/v1'))
        with self.llama_server():
            config = SetupWizard(self.paths).reconfigure(SetupOptions(provider='llamacpp'))
        self.assertEqual(config.model, 'local-model')

    def test_interactive_reconfigure_preserves_deliberate_context_capacity(self):
        with self.llama_server():
            SetupWizard(self.paths).run(SetupOptions(provider='llamacpp', context_window_tokens=16384))
        with self.llama_server(), mock.patch('builtins.input',return_value=''), redirect_stdout(io.StringIO()):
            config = SetupWizard(self.paths).reconfigure(SetupOptions(), interactive=True)
        self.assertEqual(config.context_window_tokens, 16384)

    def test_interactive_reconfigure_preserves_custom_adapter(self):
        ConfigStore(self.paths).save(Config(provider='custom', model='local-model', adapter='llamacpp',
                                           base_url='http://localhost:8080/v1', context_window_tokens=32768))
        with mock.patch('builtins.input',return_value=''), mock.patch('artificium.setup.discover_provider_models',return_value=ModelDiscovery()), redirect_stdout(io.StringIO()):
            config = SetupWizard(self.paths).reconfigure(SetupOptions(), interactive=True)
        self.assertEqual(config.adapter, 'llamacpp')

    def test_headless_setup_does_not_save_an_unauthenticated_401_connection(self):
        found = ModelDiscovery(authentication_required=True)
        with mock.patch('artificium.setup.discover_provider_models',return_value=found):
            with self.assertRaisesRegex(ValueError, 'requires an API key'):
                SetupWizard(self.paths).run(SetupOptions(provider='custom',api_url='http://localhost',
                                                        model='local',context_window_tokens=32768))
        self.assertFalse(self.paths.config.exists())

    def test_model_switch_clears_stale_generation_but_accepts_explicit_new_values(self):
        ConfigStore(self.paths).save(Config(provider='custom',base_url='http://localhost:8000/v1',
                                           model='old',reasoning_effort='high',top_k=10,vision='yes'))
        with self.llama_server():
            config = SetupWizard(self.paths).reconfigure(SetupOptions(provider='llamacpp',model='local-model',temperature=.2))
        self.assertIsNone(config.reasoning_effort)
        self.assertIsNone(config.top_k)
        self.assertEqual(config.temperature,.2)
        self.assertEqual(config.vision,'no')

    def test_multiple_models_require_a_choice_and_unknown_id_is_rejected(self):
        with mock.patch('artificium.setup.discover_provider_models',return_value=ModelDiscovery(models=('one','two'))):
            with self.assertRaisesRegex(ValueError,'Specify --model'):
                SetupWizard(self.paths).run(SetupOptions(provider='llamacpp'))
            with self.assertRaisesRegex(ValueError,'not served'):
                SetupWizard(self.paths).run(SetupOptions(provider='llamacpp',model='typo',context_window_tokens=32000))

    def test_llamacpp_wire_fields_and_reasoning_do_not_use_native_tools(self):
        config=Config(provider='llamacpp',model='local',reasoning_effort='low',repetition_penalty=1.1,top_k=30)
        request=make_engine(config,None).prepare([{'role':'user','content':'test'}])
        self.assertEqual(request.payload['repeat_penalty'],1.1)
        self.assertEqual(request.payload['reasoning_effort'],'low')
        self.assertEqual(request.payload['top_k'],30)
        self.assertNotIn('repetition_penalty',request.payload)
        self.assertNotIn('tools',request.payload)
        self.assertNotIn('Authorization',request.headers)
        with self.assertRaisesRegex(EngineError,'reasoning_budget_tokens'):
            make_engine(Config(provider='llamacpp',model='local',reasoning_budget_tokens=1024),None).prepare([])

    def test_explicit_unsupported_image_500_uses_existing_fallback_without_retry(self):
        config=Config(provider='llamacpp',model='local',vision='auto')
        image=Path(self.temp.name)/'pixel.png'
        image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='))
        failure=urllib.error.HTTPError('http://localhost',500,'error',{},io.BytesIO(b'{"error":"model does not support images"}'))
        with mock.patch('urllib.request.urlopen',side_effect=failure) as transport:
            with self.assertRaises(EngineError) as error:
                make_engine(config,None).complete([{'role':'user','content':[{'type':'artificium_image','path':str(image)}]}])
        self.assertEqual(error.exception.status,415)
        self.assertEqual(transport.call_count,1)
        self.assertTrue(image.exists())

    def test_unrelated_http500_keeps_error_and_retry_behavior(self):
        def fail(*args,**kwargs):
            raise urllib.error.HTTPError('http://localhost',500,'error',{},io.BytesIO(b'internal server error'))
        with mock.patch('urllib.request.urlopen',side_effect=fail) as transport, mock.patch('artificium.engine.time.sleep'):
            with self.assertRaises(EngineError) as error:
                make_engine(Config(provider='llamacpp',model='local'),None).complete([])
        self.assertEqual(error.exception.status,500)
        self.assertEqual(transport.call_count,3)

    def test_status_before_setup_does_not_create_runtime_or_seed_files(self):
        before={p.relative_to(self.paths.root):p.read_bytes() for p in self.paths.root.rglob('*') if p.is_file()}
        out=io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(['--root',str(self.paths.root),'status','--json']),0)
        self.assertFalse(json.loads(out.getvalue())['configured'])
        after={p.relative_to(self.paths.root):p.read_bytes() for p in self.paths.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_status_does_not_import_or_execute_mutable_scheduler(self):
        self.paths.created_tools.mkdir(parents=True)
        (self.paths.created_tools/'scheduler.py').write_text('raise RuntimeError("must not execute")')
        self.assertFalse(status_snapshot(self.paths)['configured'])

    def test_stop_refuses_to_signal_unverifiable_pid(self):
        with mock.patch('artificium.cli.process_state',return_value={'pid':123,'alive':True,'owned':None}), mock.patch('artificium.cli.os.kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'Cannot verify'):
                _stop_background(self.paths)
        kill.assert_not_called()

    def test_stop_does_not_signal_reused_pid(self):
        with mock.patch('artificium.cli.process_state',return_value={'pid':123,'alive':True,'owned':False}), mock.patch('artificium.cli.os.kill') as kill:
            self.assertFalse(_stop_background(self.paths))
        kill.assert_not_called()

    def test_clean_release_initializes_current_seeds_and_excludes_instance_data(self):
        clone=Path(self.temp.name)/'source'
        shutil.copytree(ROOT,clone,ignore=shutil.ignore_patterns('__pycache__', '.git'))
        (clone/'mind/memory/harness').mkdir(parents=True, exist_ok=True)
        (clone/'mind/self.txt').write_text('PRIVATE LIVE SELF')
        (clone/'mind/meta_memory.md').write_text('PRIVATE MEMORY MAP')
        (clone/'mind/memory/personal-secret.txt').write_text('private')
        guide='harness/tool-building-and-workspace.txt'
        (clone/'mind/memory'/guide).write_text('Stale or learned instance guide.')
        seed=clone/'artificium-code/prompts/mind-seed/memory'/guide
        seed.write_text(seed.read_text() + '\nUpdated seed for the fresh-install regression.\n')
        (clone/'artificium-code/config.json').write_text('{}')
        (clone/'artificium-code/.secrets.json').write_text('{"api_key":"private"}')
        spec=importlib.util.spec_from_file_location('build_release',ROOT/'scripts/build_release.py')
        builder=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        result=builder.build(clone,Path(self.temp.name)/'release.zip')
        with zipfile.ZipFile(result) as archive:
            names=archive.namelist()
            self.assertFalse(any(name.endswith(('config.json','.secrets.json','personal-secret.txt')) for name in names))
            self.assertFalse(any('/mind/memory/' in name or name.endswith(('/mind/self.txt', '/mind/meta_memory.md')) for name in names))
            self.assertTrue(any(name.endswith('/mind/tools/scheduler.py') for name in names))
            unpacked=Path(self.temp.name)/'unpacked'
            archive.extractall(unpacked)
        paths=Paths(next(unpacked.iterdir()))
        self.assertFalse(paths.self_file.exists())
        self.assertFalse(paths.meta_memory.exists())
        self.assertFalse(paths.memory.exists())
        prompts=PromptPack(paths)
        initialize_mind(paths, Records(paths), prompts)
        self.assertIn('persistent general-purpose agent', paths.self_file.read_text())
        self.assertEqual(paths.meta_memory.read_text(), prompts.seed('meta_memory'))
        self.assertFalse((paths.memory/'personal-secret.txt').exists())
        for item in prompts.seed_memories():
            self.assertEqual((paths.memory/item['path']).read_text().strip(), item['content'])
        self.assertIn('Updated seed for the fresh-install regression.', (paths.memory/guide).read_text())
        self.assertIn('Warning: `run_shell` blocks the life-loop', (paths.memory/guide).read_text())
