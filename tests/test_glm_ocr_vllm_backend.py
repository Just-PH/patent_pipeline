import unittest
from types import SimpleNamespace

from PIL import Image

from patent_pipeline.patent_ocr.backends.glm_ocr_vllm_backend import GlmOcrVllmBackend


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, *, messages, sampling_params):
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
            }
        )
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=f"text-{idx}")])
            for idx, _ in enumerate(messages, start=1)
        ]


class _QueuedFakeLLM:
    def __init__(self, queued_texts):
        self.calls = []
        self.queued_texts = [list(batch) for batch in queued_texts]

    def chat(self, *, messages, sampling_params):
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
            }
        )
        batch = self.queued_texts.pop(0)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for text in batch]


class GlmOcrVllmBackendTests(unittest.TestCase):
    def test_backend_batches_multiple_docs_and_builds_multimodal_messages(self):
        backend = GlmOcrVllmBackend(model_name="fake/glm-ocr")
        fake_llm = _FakeLLM()
        backend._llm = fake_llm
        backend._sampling_params_cls = lambda **kwargs: kwargs

        imgs = [Image.new("RGB", (32, 24), color=(255, 255, 255)) for _ in range(3)]
        out = backend.run_blocks_ocr(
            imgs,
            {
                "batch_size": 2,
                "prompt_text": "Text Recognition:",
                "max_new_tokens": 128,
                "resize_longest_edge": 1280,
            },
        )

        self.assertEqual(out, ["text-1", "text-2", "text-1"])
        self.assertEqual([len(call["messages"]) for call in fake_llm.calls], [2, 1])
        first_content = fake_llm.calls[0]["messages"][0][0]["content"]
        self.assertEqual(first_content[0]["type"], "image_pil")
        self.assertEqual(first_content[1]["type"], "text")
        self.assertEqual(first_content[1]["text"], "Text Recognition:")
        self.assertEqual(fake_llm.calls[0]["sampling_params"]["max_tokens"], 128)
        self.assertEqual(fake_llm.calls[0]["sampling_params"]["temperature"], 0.0)

    def test_backend_can_use_header_prompt_for_first_blocks(self):
        backend = GlmOcrVllmBackend(model_name="fake/glm-ocr")
        fake_llm = _FakeLLM()
        backend._llm = fake_llm
        backend._sampling_params_cls = lambda **kwargs: kwargs

        imgs = [Image.new("RGB", (32, 24), color=(255, 255, 255)) for _ in range(4)]
        backend.run_blocks_ocr(
            imgs,
            {
                "batch_size": 3,
                "prompt_text": "generic",
                "header_prompt_text": "header",
                "header_prompt_blocks": 2,
                "max_new_tokens": 64,
                "resize_longest_edge": 1280,
            },
        )

        prompts = [
            msg[0]["content"][1]["text"]
            for call in fake_llm.calls
            for msg in call["messages"]
        ]
        self.assertEqual(prompts, ["header", "header", "generic", "generic"])

    def test_backend_can_fallback_for_missing_classification(self):
        backend = GlmOcrVllmBackend(model_name="fake/glm-ocr")
        fake_llm = _QueuedFakeLLM(
            [
                ["header without class", "body text"],
                ["Klasse 96 g"],
            ]
        )
        backend._llm = fake_llm
        backend._sampling_params_cls = lambda **kwargs: kwargs

        imgs = [Image.new("RGB", (32, 24), color=(255, 255, 255)) for _ in range(2)]
        out = backend.run_blocks_ocr(
            imgs,
            {
                "batch_size": 8,
                "prompt_text": "generic",
                "classification_fallback_prompt_text": "class-only",
                "classification_fallback_blocks": 1,
                "classification_fallback_max_new_tokens": 64,
                "max_new_tokens": 128,
                "resize_longest_edge": 1280,
            },
        )

        self.assertEqual(len(fake_llm.calls), 2)
        self.assertEqual(
            fake_llm.calls[1]["messages"][0][0]["content"][1]["text"],
            "class-only",
        )
        self.assertTrue(out[0].startswith("Klasse 96 g"))
        self.assertIn("header without class", out[0])


if __name__ == "__main__":
    unittest.main()
