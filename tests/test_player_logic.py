import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from wolves_auto.player import AutoPlayer, is_playable_grid, side_from_initial_grid


def midgame():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[4, 1:4] = 2
    grid[0, :] = 1
    grid[1, :4] = 1
    return grid


def initial_game():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[:3, :] = 1
    grid[4, 1:4] = 2
    return grid


class PlayerSideResolutionTest(unittest.TestCase):
    def make_player(self, prompt):
        observer = SimpleNamespace(faction=None)
        with patch("wolves_auto.player.ScreenObserver", return_value=observer), \
                patch("wolves_auto.player.MouseController"):
            player = AutoPlayer(SimpleNamespace(), side="auto", side_prompt=prompt,
                                logger=lambda _line: None)
        player.observer = observer
        return player

    def test_playable_midgame_prompts_for_side_once(self):
        calls = []
        player = self.make_player(lambda obs: calls.append(obs) or "羊")
        obs = SimpleNamespace(grid=midgame(), faction=None)
        self.assertEqual(player._resolve_side(obs), "羊")
        self.assertEqual(player._resolve_side(obs), "羊")
        self.assertEqual(len(calls), 1)

    def test_invalid_detection_does_not_prompt(self):
        calls = []
        player = self.make_player(lambda obs: calls.append(obs) or "狼")
        obs = SimpleNamespace(grid=np.zeros((5, 5), dtype=np.int8), faction=None)
        self.assertIsNone(player._resolve_side(obs))
        self.assertEqual(calls, [])

    def test_initial_faction_never_prompts(self):
        calls = []
        player = self.make_player(lambda obs: calls.append(obs) or "羊")
        obs = SimpleNamespace(grid=midgame(), faction="狼")
        self.assertEqual(player._resolve_side(obs), "狼")
        self.assertEqual(calls, [])

    def test_midgame_uses_selected_side_when_turn_is_unknown(self):
        player = self.make_player(lambda _obs: "狼")
        obs = SimpleNamespace(grid=midgame(), faction=None, turn=None)
        self.assertEqual(player._should_act(obs), (True, ""))
        self.assertEqual(player.inferred_turn, "狼")

    def test_initial_game_infers_wolf_moves_first(self):
        player = self.make_player(lambda _obs: None)
        obs = SimpleNamespace(grid=initial_game(), faction="狼", turn=None)
        self.assertEqual(player._should_act(obs), (True, ""))

    def test_unknown_turn_advances_after_opponent_changes_board(self):
        player = self.make_player(lambda _obs: "狼")
        before = midgame()
        after_ours = before.copy()
        after_ours[4, 1], after_ours[3, 1] = 0, 2
        after_opponent = after_ours.copy()
        after_opponent[1, 3], after_opponent[1, 4] = 0, 1
        player.my_side = "狼"
        player.inferred_turn = "羊"
        player.awaiting = True
        player.acted_grid_sig = before.tobytes()
        player.after_my_move = after_ours
        player.await_since = 0.0
        player.await_timeout = 60.0

        obs = SimpleNamespace(grid=after_opponent, faction=None, turn=None)
        self.assertEqual(player._should_act(obs), (True, ""))
        self.assertEqual(player.inferred_turn, "狼")
        self.assertEqual(player.half_moves, 1)

    def test_illegal_detected_board_does_not_trigger_next_move(self):
        player = self.make_player(lambda _obs: "狼")
        before = midgame()
        after_ours = before.copy()
        after_ours[4, 1], after_ours[3, 1] = 0, 2
        player.my_side = "狼"
        player.inferred_turn = "羊"
        player.awaiting = True
        player.acted_grid_sig = before.tobytes()
        player.after_my_move = after_ours
        player.await_since = 0.0
        player.await_timeout = 60.0

        hallucinated = after_ours.copy()
        hallucinated[0, 4] = 1
        hallucinated[1, 4] = 1
        obs = SimpleNamespace(grid=hallucinated, faction=None, turn=None)
        self.assertEqual(player._should_act(obs), (False, "等待对手走子"))
        self.assertTrue(player.awaiting)
        self.assertEqual(player.inferred_turn, "羊")

    def test_auto_player_can_execute_after_opponent_change_without_turn_ocr(self):
        class FakeEngine:
            def __init__(self):
                self.calls = 0

            def pick_move(self, _grid, _turn, seen=None):
                self.calls += 1
                return {
                    "src": (4, 2), "dst": (3, 2), "captured": None,
                    "verdict": {"label": "和棋"}, "name": "C5-C4",
                    "next_key": ("next", self.calls), "pool_size": 3,
                }

        class FakeMouse:
            def __init__(self):
                self.moves = 0

            def move_piece(self, *_args):
                self.moves += 1
                return True

        player = self.make_player(lambda _obs: "狼")
        player.engine = FakeEngine()
        player.mouse = FakeMouse()
        before = midgame()
        obs1 = SimpleNamespace(grid=before, faction=None, turn=None, region=(0, 0, 1, 1))
        self.assertTrue(player.decide_and_act(obs1))
        self.assertEqual(player.mouse.moves, 1)

        after_opponent = before.copy()
        after_opponent[4, 2], after_opponent[3, 2] = 0, 2
        after_opponent[1, 3], after_opponent[1, 4] = 0, 1
        obs2 = SimpleNamespace(grid=after_opponent, faction=None, turn=None, region=(0, 0, 1, 1))
        self.assertTrue(player.decide_and_act(obs2))
        self.assertEqual(player.mouse.moves, 2)

    def test_position_count_allows_repeats_until_fifth_occurrence(self):
        player = self.make_player(lambda _obs: "狼")
        grid = midgame()
        self.assertEqual(player._record_position(grid, "w"), 1)
        self.assertEqual(player._record_position(grid, "w"), 1)
        for expected in range(1, 6):
            self.assertEqual(player._record_position(grid, "s"), expected)
            if expected < 5:
                self.assertEqual(player._record_position(grid, "w"), expected + 1)


class PlayableGridTest(unittest.TestCase):
    def test_requires_three_wolves_and_at_least_four_sheep(self):
        self.assertTrue(is_playable_grid(midgame()))
        grid = midgame()
        grid[4, 1] = 0
        self.assertFalse(is_playable_grid(grid))

    def test_initial_side_follows_board_orientation(self):
        self.assertEqual(side_from_initial_grid(initial_game()), "狼")
        self.assertEqual(side_from_initial_grid(np.rot90(initial_game(), 2)), "羊")


if __name__ == "__main__":
    unittest.main()
