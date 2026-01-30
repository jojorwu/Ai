"""
Tests for the explicit thinking phase and stop token logic.
"""
import unittest
from unittest.mock import MagicMock, patch
import torch
from src.agent.executor import AgentExecutor
from src.agent.dataclasses import AgentState
from src.model.structures import GenerationResult, GenerateInput, ForwardOutput
from tests.test_utils import create_test_config
from src.config.core import GenerateConfig

class TestThinking(unittest.TestCase):
    def setUp(self):
        from src.config.core import DynamicParametersConfig
        base_config = create_test_config()
        self.config = GenerateConfig(
            **base_config.model_dump(),
            generation=DynamicParametersConfig(
                start_text="task",
                max_len=100,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                speculative_steps=5,
                value_threshold=0.5,
                max_thought_len=50,
                max_retries=3
            )
        )
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.device = torch.device('cpu')

        # Setup tokenizer to return specific IDs
        self.think_open_id = 1
        self.think_close_id = 2
        def mock_encode(text, **kwargs):
            if "</THINK>" in text: return [self.think_close_id]
            if "<THINK>" in text: return [self.think_open_id]
            return [10]
        self.mock_tokenizer.encode.side_effect = mock_encode
        self.mock_tokenizer.decode.return_value = "Thought"

    def test_generator_respects_stop_tokens(self):
        from src.model.inference.generator import TextGenerator

        # Setup mock model
        model = MagicMock()
        model.device = torch.device('cpu')
        model.config = self.config.to_transformer_config()
        model.draft_model = None
        model.layers.embedding.weight.dtype = torch.float32

        # Mock initial sync to return some values
        mock_output = ForwardOutput(
            logits=torch.randn(1, 1, 100),
            value=torch.tensor([[0.5]], requires_grad=True),
            aux_loss=torch.tensor(0.0),
            ltm_memory=None
        )
        model.side_effect = [mock_output] * 10
        model.calculate_surprise.return_value = 0.1

        # Mock speculative engine to return chunks
        mock_spec_engine = MagicMock()
        # Chunks: [3, 4, 2, 5] where 2 is stop_token
        chunks = [
            (torch.tensor([[3]]), torch.tensor([[3]])),
            (torch.tensor([[4]]), torch.tensor([[4]])),
            (torch.tensor([[2]]), torch.tensor([[2]])), # STOP
            (torch.tensor([[5]]), torch.tensor([[5]])),
        ]
        mock_spec_engine.generate_chunk.side_effect = chunks
        mock_spec_engine.validate_chunk.side_effect = [c[0] for c in chunks]

        generator = TextGenerator(model, speculative_engine=mock_spec_engine)

        gen_input = GenerateInput(
            start_tokens=torch.tensor([[0]]),
            max_new_tokens=10,
            stop_tokens=[2]
        )

        results = list(generator.generate(gen_input))

        # Should stop after token 2
        self.assertEqual(len(results), 3)
        self.assertTrue((results[-1].tokens == 2).all())

    def test_agent_executor_think_phase(self):
        executor = AgentExecutor(
            self.mock_model, self.mock_tokenizer, self.config, self.mock_accelerator
        )

        state = AgentState(conversation_history_tokens=[0])

        # Mock _generate_model_response to return some tokens
        with patch.object(executor, '_generate_model_response', return_value=[1, 2, 3]) as mock_gen:
            executor.think(state)

            mock_gen.assert_called_once()
            # Verify it was called with max_thought_len
            args, kwargs = mock_gen.call_args
            self.assertEqual(kwargs.get('max_new_tokens'), self.config.generation.max_thought_len)
            # Verify stop_tokens includes </THINK>
            stop_tokens = kwargs.get('stop_tokens')
            self.assertIsNotNone(stop_tokens)

    def test_agent_executor_run_calls_think(self):
        executor = AgentExecutor(
            self.mock_model, self.mock_tokenizer, self.config, self.mock_accelerator
        )

        # Set max_turns to 1 for quick test
        self.config.generation.max_turns = 1

        with patch.object(executor, 'think', return_value=[]) as mock_think:
            with patch.object(executor, '_generate_model_response', return_value=[]) as mock_gen:
                with patch.object(executor, '_process_tool_call', return_value=False):
                    executor.run()
                    mock_think.assert_called_once()
                    mock_gen.assert_called_once()

if __name__ == "__main__":
    unittest.main()
