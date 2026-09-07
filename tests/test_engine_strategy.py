import unittest

from wolves_auto.engine import (Engine, board_from_str, shortlist_optimal_moves,
                                 verdict_result_code)
from wolves_eat_sheep_hard_solve import hard_solve_fast as hsf


def move(name, rank, dist, key=None):
    return {
        "name": name,
        "rank": rank,
        "verdict": {"dist": dist},
        "next_key": key or name,
    }


def result_move(name, rank, dist, result):
    item = move(name, rank, dist)
    item["verdict"]["result"] = result
    return item


class ShortlistOptimalMovesTest(unittest.TestCase):
    def test_winning_pool_uses_three_shortest_moves(self):
        ranked = [
            move("a", 2, 8), move("b", 2, 3), move("c", 2, 5),
            move("d", 2, 4), move("draw", 1, 0),
        ]
        self.assertEqual(
            [m["name"] for m in shortlist_optimal_moves(ranked)],
            ["b", "d", "c"],
        )

    def test_seen_positions_are_removed_before_limit(self):
        ranked = [
            move("repeat-1", 2, 1), move("repeat-2", 2, 2),
            move("fresh-1", 2, 3), move("fresh-2", 2, 4),
        ]
        pool = shortlist_optimal_moves(ranked, seen={"repeat-1", "repeat-2"})
        self.assertEqual([m["name"] for m in pool], ["fresh-1", "fresh-2"])

    def test_seen_positions_are_allowed_when_no_fresh_move_exists(self):
        ranked = [move("repeat-1", 1, 0), move("repeat-2", 1, 0)]
        self.assertEqual(
            [m["name"] for m in shortlist_optimal_moves(
                ranked, seen={"repeat-1", "repeat-2"})],
            ["repeat-1", "repeat-2"],
        )

    def test_losing_pool_keeps_three_longest_defenses(self):
        ranked = [move("a", 0, 4), move("b", 0, 9),
                  move("c", 0, 7), move("d", 0, 5)]
        self.assertEqual(
            [m["name"] for m in shortlist_optimal_moves(ranked)],
            ["b", "c", "d"],
        )

    def test_lower_result_tier_never_enters_pool(self):
        ranked = [move("win", 2, 9), move("draw", 1, 0), move("loss", 0, 20)]
        self.assertEqual(
            [m["name"] for m in shortlist_optimal_moves(ranked)],
            ["win"],
        )

    def test_required_tablebase_result_excludes_other_outcomes(self):
        ranked = [
            result_move("draw", 1, 0, hsf.DRAW),
            result_move("wolf-win", 2, 4, hsf.WOLF_WIN),
            result_move("sheep-win", 0, 8, hsf.SHEEP_WIN),
        ]
        pool = shortlist_optimal_moves(ranked, required_result=hsf.WOLF_WIN)
        self.assertEqual([item["name"] for item in pool], ["wolf-win"])

    def test_terminal_verdict_uses_same_result_code(self):
        self.assertEqual(verdict_result_code({"result": hsf.SHEEP_WIN}), hsf.SHEEP_WIN)
        self.assertEqual(
            verdict_result_code({"result": None, "terminal": ("狼胜", "吃净")}),
            hsf.WOLF_WIN,
        )

    def test_missing_required_result_returns_no_pool(self):
        ranked = [result_move("draw", 1, 0, hsf.DRAW)]
        self.assertEqual(shortlist_optimal_moves(ranked, required_result=hsf.WOLF_WIN), [])

    def test_reported_draw_position_never_selects_losing_b5_b4(self):
        grid = board_from_str("sssss" "s..ss" "ssss." "...W." ".W.W.")
        engine = Engine()
        for _ in range(20):
            move = engine.pick_move(grid, "w")
            self.assertIsNotNone(move)
            self.assertNotEqual(move["name"], "B5-B4")
            self.assertEqual(move["verdict"]["result"], hsf.DRAW)


if __name__ == "__main__":
    unittest.main()
