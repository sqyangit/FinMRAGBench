"""Regression checks for entry points and agent control flow without GPU models.

Run with: python -m unittest discover -s tests
"""

from contextlib import redirect_stdout
import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def module_stub(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def load_module(relative_path, stubs=None):
    spec = importlib.util.spec_from_file_location(
        '_test_' + Path(relative_path).stem, ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs or {}), patch.object(
        sys, 'path', [str((ROOT / relative_path).parent), *sys.path],
    ):
        spec.loader.exec_module(module)
    return module


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def agent(self, filename, class_name, replies, **kwargs):
        client = Mock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
            for text in replies
        ]
        checker = Mock(return_value=Mock(generate=Mock(return_value='{"score":"8"}')))
        module = load_module(filename, {
            'openai': module_stub('openai', OpenAI=Mock(return_value=client)),
            'transformers': module_stub('transformers', AutoModel=Mock()),
            'llm_local': module_stub('llm_local', VLM=checker),
        })
        instance = getattr(module, class_name)(**kwargs)
        instance.search = Mock(return_value={'source_nodes': []})
        return instance, client, checker

    def test_actions_inside_reasoning_are_not_executed(self):
        for filename, class_name, tag in [
            ('FinMRAGAgent/tooluse_agent.py', 'ToolAgent', 'think'),
            ('FinMRAGAgent/react_agent.py', 'ReactAgent', 'thinking'),
        ]:
            with self.subTest(agent=class_name):
                agent, _, _ = self.agent(filename, class_name, [
                    f'<{tag}>Example: <search>invalid</search></{tag}><answer>42</answer>',
                ])
                self.assertEqual(agent.run('question')[0], '42')
                agent.search.assert_not_called()

    def test_invalid_queries_never_reach_search(self):
        for body in ['invalid JSON', '[]', '{}', '{"query":null}']:
            for filename, class_name in [
                ('FinMRAGAgent/tooluse_agent.py', 'ToolAgent'),
                ('FinMRAGAgent/react_agent.py', 'ReactAgent'),
            ]:
                with self.subTest(agent=class_name, body=body):
                    agent, _, _ = self.agent(filename, class_name, [f'<search>{body}</search>'])
                    with self.assertRaises(ValueError):
                        agent.run('question')
                    agent.search.assert_not_called()

    def test_step_limit_allows_a_final_answer_but_no_extra_search(self):
        for filename, class_name in [
            ('FinMRAGAgent/tooluse_agent.py', 'ToolAgent'),
            ('FinMRAGAgent/react_agent.py', 'ReactAgent'),
        ]:
            for final_reply in ['<answer>42</answer>', '<search>{"query":"again"}</search>']:
                with self.subTest(agent=class_name, final_reply=final_reply):
                    agent, client, _ = self.agent(filename, class_name, [
                        '<search>{"query":"revenue"}</search>', final_reply,
                    ], max_steps=1)
                    answer, pages, history = agent.run('question', return_history=True)
                    self.assertEqual(client.chat.completions.create.call_count, 2)
                    self.assertEqual(agent.search.call_count, 1)
                    self.assertEqual(pages, [])
                    if final_reply.startswith('<answer>'):
                        self.assertEqual(answer, '42')
                    else:
                        self.assertEqual(history[-1]['reason'], 'max_steps_exceeded')

    def test_selected_dataset_checker_and_images_are_used(self):
        for filename, class_name in [
            ('FinMRAGAgent/tooluse_agent.py', 'ToolAgent'),
            ('FinMRAGAgent/react_agent.py', 'ReactAgent'),
        ]:
            with self.subTest(agent=class_name), tempfile.TemporaryDirectory() as folder:
                Image.new('RGB', (16, 16)).save(Path(folder) / 'page.jpg')
                agent, _, checker = self.agent(filename, class_name, [
                    '<search>{"query":"revenue"}</search>', '<answer>42</answer>',
                ], image_dir=folder, check_model_name='custom-checker', max_steps=1)
                agent.min_pixels = 1
                nodes = [{'node': {'metadata': {'file_name': 'page.jpg'}}}] * 2
                agent.search.return_value = {'source_nodes': nodes}
                agent.reranker.compute_score.return_value = [0.9, 0.8]
                answer, pages = agent.run('question')
                self.assertEqual((answer, pages), ('42', ['page.jpg']))
                self.assertEqual(agent.image_path, [str(Path(folder) / 'page.jpg')])
                self.assertIsInstance(agent.image_input[0], Image.Image)
                checker.assert_called_with('custom-checker', agent.check_url)
                pairs = agent.reranker.compute_score.call_args.args[0]
                self.assertEqual(pairs[0][1], str(Path(folder) / 'page.jpg'))


class EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'FinMRAGAgent'))
        cls.entry = load_module('FinMRAGAgent/eval.py')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'FinMRAGAgent'))

    def test_dispatch_passes_dataset_and_agent_settings(self):
        for experiment, module_name, class_name in [
            ('tooluse', 'tooluse_agent', 'ToolAgent'),
            ('react', 'react_agent', 'ReactAgent'),
        ]:
            with self.subTest(experiment=experiment):
                constructor = Mock()
                with patch.dict(sys.modules, {module_name: module_stub(module_name, **{class_name: constructor})}):
                    model = self.entry.MMRAG(experiment_type=experiment, dataset='CustomReports',
                                             check_vlm='custom-checker', max_steps=3)
                kwargs = constructor.call_args.kwargs
                self.assertEqual(kwargs['image_dir'], model.img_dir)
                self.assertEqual(kwargs['check_model_name'], 'custom-checker')
                self.assertEqual(kwargs['max_steps'], 3)
                self.assertEqual(kwargs['base_url'], 'http://127.0.0.1:8001/v1')
                self.assertIsNone(model.vlm)

    def test_failure_keeps_retrieved_evidence_and_resume_skips_saved_ids(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(self.entry, 'PROJECT_ROOT', folder):
            annotations = Path(folder) / 'data_bench'
            annotations.mkdir()
            (annotations / 'numerical_reasoning.json').write_text(
                json.dumps({'result': [{'id': 1}, {'id': 2}]}), encoding='utf-8',
            )
            llm = module_stub('llms.llm', LLM=Mock())
            with patch.dict(sys.modules, {'llms.llm': llm}):
                model = self.entry.MMRAG(workers_num=2)

            def infer(sample):
                sample['recall_results'] = {'source_nodes': [{'page': 'page.jpg'}]}
                if sample['id'] == 1:
                    raise RuntimeError('mock generation failure')
                return dict(sample, response='ok')

            model.eval_func = Mock(side_effect=infer)
            output = Path(model.eval_dataset())
            rows = {row['id']: row for row in map(json.loads, output.read_text(encoding='utf-8').splitlines())}
            self.assertEqual(set(rows), {1, 2})
            self.assertEqual(rows[1]['status'], 'error')
            self.assertEqual(rows[1]['recall_results']['source_nodes'], [{'page': 'page.jpg'}])
            self.assertEqual(rows[2]['response'], 'ok')
            model.eval_func.reset_mock()
            model.eval_dataset()
            model.eval_func.assert_not_called()

    def test_agent_failure_keeps_completed_search_steps(self):
        model = self.entry.MMRAG.__new__(self.entry.MMRAG)
        history = [{'query': 'q'}, {'selected_images': ['page.jpg']}]
        model.agents = SimpleNamespace(last_history=history, run=Mock(side_effect=ValueError('invalid action')))
        model._recall_results = lambda pages: {'pages': pages}
        result = model.tooluse({'question': 'q'})
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['history'], history)
        self.assertEqual(result['recall_results']['pages'], ['page.jpg'])


class ModelRoutingTests(unittest.TestCase):
    def test_custom_served_names_and_unknown_models(self):
        # Compile the wrapper classes themselves; importing GPU backends is unnecessary.
        for filename, class_name, backend in [
            ('FinMRAGAgent/llms/llm.py', 'LLM', 'Qwen_VL_2_5'),
            ('inference/llm_local.py', 'VLM', 'Qwen_VLM_2_5'),
        ]:
            with self.subTest(wrapper=class_name):
                tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
                node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
                local = Mock(return_value=Mock(generate=Mock(return_value='answer')))
                remote = Mock()
                namespace = {backend: local, 'GPT_LOCAL': remote}
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
                wrapper = namespace[class_name]('our-custom-agent', 'http://localhost:8001/v1')
                self.assertEqual(wrapper.generate(query='q', image=[]), 'answer')
                local.assert_called_with('our-custom-agent', 'http://localhost:8001/v1')
                remote.assert_not_called()
                namespace[class_name]('o3')
                remote.assert_called_with('o3')
                with self.assertRaises(ValueError):
                    namespace[class_name]('unknown-model')


class ViDoRAGTests(unittest.TestCase):
    def module(self):
        names = ['seeker_prompt'] + [f'{role}_prompt_{task}' for role in ['inspector', 'answer']
                                   for task in ['numerical', 'table', 'chart', 'knowledge', 'fact']]
        return load_module('FinMRAGAgent/vidorag_agents.py', {
            'agent.agent_prompt': module_stub('agent.agent_prompt', **dict.fromkeys(names, 'test')),
            'agent.map_dict': module_stub('agent.map_dict', arrangement_map_dict={1: [1, 1]},
                                         page_map_dict_normal={1: 'page'}, page_map_dict={1: 'page'}),
            'json_repair': module_stub('json_repair', repair_json=lambda text, **kwargs: text),
        })

    def test_reasoning_rounds_are_bounded(self):
        module = self.module()
        agent = module.ViDoRAG_Agents(Mock(), max_rounds=2)
        agent.seeker = Mock(run=Mock(return_value=(['page.jpg'], 'summary', 'reason')))
        agent.inspector = Mock(run=Mock(return_value=('seeker', 'more evidence', [])))
        with self.assertRaisesRegex(RuntimeError, 'exceeded 2'):
            agent.run_agent('question', ['page.jpg'], 'numerical_reasoning')
        self.assertEqual(agent.inspector.run.call_count, 2)

    def test_negative_page_indices_are_rejected(self):
        module = self.module()
        seeker = module.Seeker(Mock(generate=Mock(return_value=json.dumps({
            'reason': 'test', 'summary': 'test', 'choice': [-1],
        }))))
        seeker.seeker_multi_image = True
        with self.assertRaisesRegex(Exception, 'time out'):
            seeker.run('question', ['page.jpg'])


if __name__ == '__main__':
    unittest.main()
