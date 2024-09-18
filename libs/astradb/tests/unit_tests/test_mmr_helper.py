from langchain_core.documents import Document

from langchain_astradb.utils.mmr_traversal import MmrHelper

IDS = {
    "-1",
    "-2",
    "-3",
    "-4",
    "-5",
    "+1",
    "+2",
    "+3",
    "+4",
    "+5",
}


class TestMmrHelper:
    def test_mmr_helper_functional(self) -> None:
        helper = MmrHelper(k=3, query_embedding=[6, 5], lambda_mult=0.5)

        assert len(list(helper.candidate_ids())) == 0

        helper.add_candidates({"-1": (Document(page_content="-1"), [3, 5])})
        helper.add_candidates({"-2": (Document(page_content="-2"), [3, 5])})
        helper.add_candidates({"-3": (Document(page_content="-3"), [2, 6])})
        helper.add_candidates({"-4": (Document(page_content="-4"), [1, 6])})
        helper.add_candidates({"-5": (Document(page_content="-5"), [0, 6])})

        assert len(list(helper.candidate_ids())) == 5

        helper.add_candidates({"+1": (Document(page_content="+1"), [5, 3])})
        helper.add_candidates({"+2": (Document(page_content="+2"), [5, 3])})
        helper.add_candidates({"+3": (Document(page_content="+3"), [6, 2])})
        helper.add_candidates({"+4": (Document(page_content="+4"), [6, 1])})
        helper.add_candidates({"+5": (Document(page_content="+5"), [6, 0])})

        assert len(list(helper.candidate_ids())) == 10

        for idx in range(3):
            best_id = helper.pop_best()
            assert best_id in IDS
            assert len(list(helper.candidate_ids())) == 9 - idx
            assert best_id not in helper.candidate_ids()

    def test_mmr_helper_max_diversity(self) -> None:
        helper = MmrHelper(k=2, query_embedding=[6, 5], lambda_mult=0)
        helper.add_candidates({"-1": (Document(page_content="-1"), [3, 5])})
        helper.add_candidates({"-2": (Document(page_content="-2"), [3, 5])})
        helper.add_candidates({"-3": (Document(page_content="-3"), [2, 6])})
        helper.add_candidates({"-4": (Document(page_content="-4"), [1, 6])})
        helper.add_candidates({"-5": (Document(page_content="-5"), [0, 6])})

        best = {helper.pop_best(), helper.pop_best()}
        assert best == {"-1", "-5"}

    def test_mmr_helper_max_similarity(self) -> None:
        helper = MmrHelper(k=2, query_embedding=[6, 5], lambda_mult=1)
        helper.add_candidates({"-1": (Document(page_content="-1"), [3, 5])})
        helper.add_candidates({"-2": (Document(page_content="-2"), [3, 5])})
        helper.add_candidates({"-3": (Document(page_content="-3"), [2, 6])})
        helper.add_candidates({"-4": (Document(page_content="-4"), [1, 6])})
        helper.add_candidates({"-5": (Document(page_content="-5"), [0, 6])})

        best = {helper.pop_best(), helper.pop_best()}
        assert best == {"-1", "-2"}
