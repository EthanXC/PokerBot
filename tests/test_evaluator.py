"""Sanity tests for the hand evaluator.

Each test is a hand of known category vs another hand of known category;
we just verify the ordering. Run as:

    python -m unittest tests.test_evaluator -v
"""
from __future__ import annotations

import unittest

from pokerbot.core import Card, HandEvaluator


def C(s: str) -> Card:
    """Tiny helper: 'AS' -> Card('A','S'), '10H' -> Card('10','H')."""
    if s.startswith("10"):
        return Card("10", s[2])
    return Card(s[0], s[1])


def cards(*ss: str) -> list[Card]:
    return [C(s) for s in ss]


class EvaluatorOrderingTest(unittest.TestCase):
    def assertBeats(self, winner_5, loser_5, msg=""):
        ws = HandEvaluator.score_five(winner_5)
        ls = HandEvaluator.score_five(loser_5)
        self.assertGreater(
            ws, ls, f"{msg}: expected {winner_5} > {loser_5}, got {ws} vs {ls}"
        )

    def test_straight_flush_beats_quads(self):
        sf = cards("9H", "8H", "7H", "6H", "5H")
        quads = cards("AS", "AH", "AC", "AD", "KH")
        self.assertBeats(sf, quads, "straight flush beats quads")

    def test_quads_beats_full_house(self):
        q = cards("KS", "KH", "KC", "KD", "2H")
        fh = cards("AS", "AH", "AC", "KH", "KD")
        self.assertBeats(q, fh)

    def test_full_house_beats_flush(self):
        fh = cards("9S", "9H", "9C", "2S", "2D")
        fl = cards("AS", "JS", "8S", "5S", "2S")
        self.assertBeats(fh, fl)

    def test_flush_beats_straight(self):
        fl = cards("AS", "JS", "8S", "5S", "2S")
        st = cards("9H", "8C", "7D", "6S", "5H")
        self.assertBeats(fl, st)

    def test_wheel_straight(self):
        # A-5 wheel; 5-high straight, NOT ace-high
        wheel = cards("AS", "2H", "3D", "4C", "5S")
        five_high = cards("5S", "4H", "3D", "2C", "AH")
        # they're the same hand
        self.assertEqual(HandEvaluator.score_five(wheel), HandEvaluator.score_five(five_high))
        # six-high beats five-high (wheel)
        six_high = cards("6S", "5H", "4D", "3C", "2H")
        self.assertBeats(six_high, wheel, "6-high straight beats wheel")

    def test_higher_pair_beats_lower(self):
        kk = cards("KS", "KH", "5C", "3D", "2H")
        qq = cards("QS", "QH", "AC", "JD", "9H")  # higher kickers
        self.assertBeats(kk, qq, "KK > QQ even with worse kickers")

    def test_kicker_breaks_tie(self):
        ace_king = cards("AS", "AH", "KC", "5D", "2H")
        ace_queen = cards("AC", "AD", "QH", "JS", "9C")
        self.assertBeats(ace_king, ace_queen)

    def test_score_seven_picks_best_five(self):
        # 5 hearts on the board => flush
        hand = cards("AS", "KS", "9H", "8H", "7H", "6H", "5H")
        score = HandEvaluator.score_seven(hand)
        # straight flush 9-high
        self.assertEqual(score[0], 8)  # CAT_STRAIGHT_FLUSH
        self.assertEqual(score[1], 9)


class ShowdownTest(unittest.TestCase):
    def test_simple_showdown(self):
        from pokerbot.core import Player

        p1 = Player(seat=1, hole=(C("AS"), C("AH")))
        p2 = Player(seat=2, hole=(C("KS"), C("KH")))
        board = cards("2C", "5D", "9H", "JS", "3C")
        winners, _ = HandEvaluator.showdown([p1, p2], board)
        self.assertEqual(winners, [0])

    def test_chop(self):
        from pokerbot.core import Player

        # Both players play the board's straight: T-J-Q-K-A
        p1 = Player(seat=1, hole=(C("2S"), C("3H")))
        p2 = Player(seat=2, hole=(C("4C"), C("5D")))
        board = cards("10S", "JD", "QH", "KS", "AC")
        winners, _ = HandEvaluator.showdown([p1, p2], board)
        self.assertEqual(set(winners), {0, 1})


if __name__ == "__main__":
    unittest.main()
