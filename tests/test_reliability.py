import unittest
import pandas as pd
from src.agent import parse_response
from src.audit_results import vote, audit
from src.metrics import expected_calibration_error
from src.orchestrator import run_debate_on_question


class ReliabilityTests(unittest.TestCase):
    def test_valid_response(self):
        self.assertEqual(parse_response('REASONING: test\nANSWER: B\nCONFIDENCE: 100')['confidence'], 100)

    def test_invalid_responses_fail_instead_of_random_answers(self):
        for text in ['', 'ANSWER: A', 'ANSWER: Apple\nCONFIDENCE: 90', 'ANSWER: A\nCONFIDENCE: 1000', 'ANSWER: B\nCONFIDENCE: 101']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response(text)

    def test_ties_abstain_regardless_of_agent_order(self):
        data = pd.DataFrame({'answer': ['A', 'B', 'C'], 'confidence': [90, 90, 90]})
        for weighted in [True, False]:
            self.assertIsNone(vote(data, weighted))
            self.assertIsNone(vote(data.iloc[::-1], weighted))

    def test_ece_includes_zero_and_hundred(self):
        data = pd.DataFrame({'confidence': [0, 100], 'is_correct': [False, True]})
        self.assertAlmostEqual(expected_calibration_error(data), 0)

    def test_rounds_are_synchronous(self):
        class Fake:
            persona_key = 'test'
            def __init__(self, name): self.name = name
            def respond(self, question, options, peers, round_num):
                if round_num == 2:
                    assert len(peers) == 1
                    assert all(r['confidence'] == 10 for r in peers.values())
                return dict(answer='A', confidence=10 * round_num, reasoning='test', parsed_ok=True)
        logs, _ = run_debate_on_question([Fake('a'), Fake('b')], dict(id='q', question='?', options=dict(A='a', B='b', C='c', D='d'), answer='A'))
        self.assertEqual(len(logs), 4)


if __name__ == '__main__': unittest.main()
