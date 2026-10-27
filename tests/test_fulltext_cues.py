import unittest

from scripts.paper_pipeline.fulltext_cues import cue_windows


class FullTextCueTests(unittest.TestCase):
    def make_page(self, texts):
        value, blocks = "", []
        for text in texts:
            start = len(value)
            value += text + "\n\n"
            blocks.append({"text": text, "start": start, "end": start + len(text)})
        return {"page": 1, "text": value, "blocks": blocks}

    def test_windows_preserve_exact_page_offsets_and_skip_old_selection(self):
        first = "We propose BlueNet, a new approach for image classification."
        second = "We evaluate BlueNet on the CIFAR-10 dataset for the held-out experiments."
        page = self.make_page([first, second])
        paper = {"id": "paper:test", "pdf": "a.pdf", "pdf_sha256": "abc"}
        parsed = {"paper_id": paper["id"], "pdf_sha256": "abc", "pages": [page]}
        selected = {"passages": [{"page": 1, "start": 0, "end": len(first)}]}
        result = cue_windows(paper, parsed, selected)
        self.assertEqual(len(result["passages"]), 1)
        item = result["passages"][0]
        self.assertEqual(item["text"], second)
        self.assertEqual(page["text"][item["start"]:item["end"]], item["text"])
        self.assertIn("dataset_or_experiment", item["cue_types"])

    def test_reference_section_is_not_misread_as_experiment(self):
        page = self.make_page(["References", "[1] A paper on the training dataset and benchmark.",
                               "Appendix A", "We evaluate our model on the Example dataset in the appendix."])
        paper = {"id": "paper:test", "pdf": "a.pdf", "pdf_sha256": "abc"}
        parsed = {"paper_id": paper["id"], "pdf_sha256": "abc", "pages": [page]}
        result = cue_windows(paper, parsed, {"passages": []})
        self.assertEqual(len(result["passages"]), 1)
        self.assertIn("Example dataset", result["passages"][0]["text"])

    def test_wrong_source_identity_or_invalid_block_is_rejected(self):
        page = self.make_page(["We evaluate the BlueNet model on Example dataset."])
        paper = {"id": "paper:test", "pdf": "a.pdf", "pdf_sha256": "abc"}
        parsed = {"paper_id": "paper:other", "pdf_sha256": "abc", "pages": [page]}
        with self.assertRaises(ValueError):
            cue_windows(paper, parsed, {"passages": []})
        parsed["paper_id"] = "paper:test"
        page["blocks"][0]["text"] = "wrong"
        with self.assertRaises(ValueError):
            cue_windows(paper, parsed, {"passages": []})


if __name__ == "__main__":
    unittest.main()
