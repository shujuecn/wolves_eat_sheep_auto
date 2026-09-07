"""自动下棋：屏幕观测 -> 表库最优解 -> 鼠标走子 的闭环。

流程：
  1. 观测窗口得到稳定 5x5 局面 + 走棋方（紫色提示区红/白字）；
  2. 完整初始局自动判定阵营；残局通过回调询问用户阵营；
  3. 终局检测（羊<4 / 三狼全堵死 / 重复局面 5 次 / 步数上限）→ 报告结果，等待下一局；
  4. 轮到我方且局面稳定时：表库选步（最优档前三条随机）
     → 输出分析 → 鼠标执行；
  5. 我方走完后进入等待状态，直到观测到对手造成的再次局面变化，
     或等待超时后才重新评估。

demo 模式只打印建议走法、不点击；每个局面只建议一次。
"""
from __future__ import annotations

import time

import numpy as np

from .engine import (EMPTY, Engine, MAX_MOVES,
                     cell_name, game_winner_grid)
from .mousectl import MouseController
from .screen import ScreenObserver
from .screen import SHEEP as CV_SHEEP, WOLF as CV_WOLF

SIDE_CN2EN = {"狼": "w", "羊": "s"}
SYM = {EMPTY: ".", CV_SHEEP: "s", CV_WOLF: "W"}


def is_initial_grid(grid: np.ndarray) -> bool:
    """标准初始局面（狼在最下排中央 或 180° 旋转），且恰有 3狼15羊。"""
    g = np.asarray(grid)
    if int((g == CV_WOLF).sum()) != 3 or int((g == CV_SHEEP).sum()) != 15:
        return False
    bottom = bool((g[4, 1:4] == CV_WOLF).all() and (g[0:3, :] == CV_SHEEP).all())
    top = bool((g[0, 1:4] == CV_WOLF).all() and (g[2:5, :] == CV_SHEEP).all())
    return bottom or top


def side_from_initial_grid(grid: np.ndarray) -> str | None:
    """按完整初始局朝向返回用户执棋方。"""
    g = np.asarray(grid)
    if not is_initial_grid(g):
        return None
    return "狼" if bool((g[4, 1:4] == CV_WOLF).all()) else "羊"


def is_playable_grid(grid: np.ndarray) -> bool:
    """是否为可询问执棋方的有效对局局面。"""
    g = np.asarray(grid)
    return int((g == CV_WOLF).sum()) == 3 and 4 <= int((g == CV_SHEEP).sum()) <= 15


class AutoPlayer:
    def __init__(self, engine: Engine, side: str = "auto", demo: bool = False,
                 region: tuple[int, int, int, int] | None = None,
                 post_move_delay: float = 1.2,
                 await_timeout: float = 60.0,
                 max_half_moves: int | None = MAX_MOVES,
                 side_prompt=None, logger=print, on_observation=None):
        self.engine = engine
        self.side_arg = side                  # "狼" | "羊" | "auto"
        self.observer = ScreenObserver(region=region, logger=logger)
        self.log = logger
        self.side_prompt = side_prompt
        self.on_observation = on_observation
        self.mouse = MouseController(demo=demo, logger=logger)
        self.demo = demo
        self.post_move_delay = post_move_delay
        self.await_timeout = await_timeout
        self.max_half_moves = max_half_moves

        # ---- 运行时状态 ----
        self.my_side: str | None = None       # "狼" / "羊"
        self.position_counts: dict[tuple, int] = {}
        self._last_position_key = None
        self.awaiting = False                 # 我方已走、等对手
        self.await_since = 0.0
        self.after_my_move: np.ndarray | None = None
        self.acted_grid_sig: bytes | None = None   # 出过手局面的签名
        self.half_moves = 0                   # 本局双方合计步数
        self.games = 0
        self._last_initial: np.ndarray | None = None
        self._side_prompted_sig: bytes | None = None
        self._terminal_sig: bytes | None = None
        self.inferred_turn: str | None = None
        self.opponent_grid_sig: bytes | None = None
        self._invalid_grid_sig: bytes | None = None

    # ------------------------------------------------------------------
    def _resolve_side(self, obs):
        if self.my_side is None:
            if self.side_arg in SIDE_CN2EN:
                self.my_side = self.side_arg
            else:
                detected_side = side_from_initial_grid(obs.grid) or obs.faction
                if detected_side:
                    self.my_side = detected_side
                    self.observer.faction = detected_side
                    self.log(f"阵营 | 已从完整棋局识别为{self.my_side}方")
                elif is_playable_grid(obs.grid) and self.side_prompt is not None:
                    sig = np.asarray(obs.grid, dtype=np.int8).tobytes()
                    if sig != self._side_prompted_sig:
                        self._side_prompted_sig = sig
                        chosen = self.side_prompt(obs)
                        if chosen in SIDE_CN2EN:
                            self.my_side = chosen
                            self.observer.faction = chosen
                            self.log(f"阵营 | 残局已选择{chosen}方")

        if self.my_side is not None and self.inferred_turn is None:
            if is_initial_grid(obs.grid):
                self.inferred_turn = "狼"
            else:
                observed_turn = getattr(obs, "turn", None)
                self.inferred_turn = observed_turn or self.my_side
                if observed_turn is None:
                    self.log(f"回合 | 提示未识别，按{self.my_side}方当前行棋接管残局")
            if self.inferred_turn != self.my_side:
                self.opponent_grid_sig = np.asarray(obs.grid, dtype=np.int8).tobytes()
        return self.my_side

    def _clear_game(self):
        """清空本局运行时状态（终局后/新局前）。"""
        self.position_counts.clear()
        self._last_position_key = None
        self.awaiting = False
        self.after_my_move = None
        self.acted_grid_sig = None
        self.half_moves = 0
        self._side_prompted_sig = None
        self.inferred_turn = None
        self.opponent_grid_sig = None
        self._invalid_grid_sig = None
        if self.side_arg not in SIDE_CN2EN:
            self.my_side = None               # auto：随新局重新判定
            self.observer.faction = None

    def _start_new_game(self, reason: str):
        self.games += 1
        self._clear_game()
        self._terminal_sig = None
        self.log(f"对局 | 第 {self.games} 局开始（{reason}）")

    # ------------------------------------------------------------------
    def _should_act(self, obs) -> tuple[bool, str]:
        if self._resolve_side(obs) is None:
            return False, "等待确认执棋方"
        if self.awaiting:
            sig = np.asarray(obs.grid, dtype=np.int8).tobytes()
            still_before = sig == self.acted_grid_sig
            still_after = (self.after_my_move is not None
                           and np.array_equal(obs.grid, self.after_my_move))
            waited = time.perf_counter() - self.await_since
            opponent = "羊" if self.my_side == "狼" else "狼"
            opponent_moved = (not still_before and not still_after and
                              self._is_legal_successor(self.after_my_move,
                                                       obs.grid, opponent))
            if opponent_moved:
                self.awaiting = False         # 局面再次变化，对手已经走子
                self.inferred_turn = self.my_side
                self.opponent_grid_sig = None
                self._invalid_grid_sig = None
                self.half_moves += 1
            elif waited > self.await_timeout:
                if still_before:
                    self.awaiting = False
                    self.acted_grid_sig = None
                    self.inferred_turn = self.my_side
                    self.log("执行 | 棋盘未变化，重新尝试本步")
                else:
                    self.await_since = time.perf_counter()
                    self.log("等待 | 对手尚未行棋")
                    return False, "等待对手走子"
            else:
                if not still_before and not still_after and sig != self._invalid_grid_sig:
                    self._invalid_grid_sig = sig
                    self.log("检测 | 忽略不符合合法行棋的棋盘变化")
                return False, "等待对手走子"

        sig = np.asarray(obs.grid, dtype=np.int8).tobytes()
        if self.inferred_turn != self.my_side:
            if self.opponent_grid_sig is None:
                self.opponent_grid_sig = sig
            elif sig != self.opponent_grid_sig:
                self.inferred_turn = self.my_side
                self.opponent_grid_sig = None
                self.half_moves += 1
        if self.inferred_turn != self.my_side:
            return False, f"轮到{self.inferred_turn or '对手'}方"
        return True, ""

    @staticmethod
    def _is_legal_successor(before, observed, side: str) -> bool:
        """确认观测到的变化确实是对手从上一稳定局面走的一步。"""
        from .engine import game_from_grid

        if before is None:
            return False
        source = np.asarray(before, dtype=np.int8)
        target = np.asarray(observed, dtype=np.int8)
        game = game_from_grid([[int(v) for v in row] for row in source],
                              SIDE_CN2EN[side])
        for row in range(5):
            for col in range(5):
                if game.board[row][col] != game.turn:
                    continue
                for move in game.legal_moves_from((row, col)):
                    candidate = source.copy()
                    piece = int(candidate[row, col])
                    candidate[row, col] = EMPTY
                    if move.captured:
                        candidate[move.captured[0], move.captured[1]] = EMPTY
                    candidate[move.destination[0], move.destination[1]] = piece
                    if np.array_equal(candidate, target):
                        return True
        return False

    # ------------------------------------------------------------------
    def decide_and_act(self, obs) -> bool:
        """对一次观测做决策并（非 demo 时）执行走子。返回是否出手。"""
        grid_list = [[int(v) for v in row] for row in obs.grid]
        sig = np.asarray(obs.grid, dtype=np.int8).tobytes()

        # 终局检测（两种胜负均与回合无关）
        winner = game_winner_grid(grid_list, "w")
        if winner is not None:
            if self._terminal_sig == sig:
                return False
            self._terminal_sig = sig
            result, reason = winner
            mine = (("狼" in result) == (self.my_side == "狼")) \
                if self.my_side else None
            tag = {True: "，我方胜", False: "，我方负"}.get(mine, "")
            self.log(f"终局 | {result}（{reason}{tag}）")
            self._clear_game()
            self._last_initial = None
            self.log("等待 | 检测到新局后自动继续")
            return False
        if self._terminal_sig == sig:
            return False
        self._terminal_sig = None

        act, why = self._should_act(obs)
        turn_cn = self.inferred_turn or getattr(obs, "turn", None) or self.my_side
        turn_en = SIDE_CN2EN.get(turn_cn)
        count = self._record_position(obs.grid, turn_en) if turn_en else 0
        if count >= 5:
            self._terminal_sig = sig
            self.log("终局 | 和棋（重复局面达到 5 次）")
            self._clear_game()
            self._last_initial = None
            return False
        if self.max_half_moves and self.half_moves >= self.max_half_moves:
            self._terminal_sig = sig
            self.log(f"终局 | 和棋（达到 {self.max_half_moves} 步上限）")
            self._clear_game()
            self._last_initial = None
            return False
        if not act:
            return False

        if self.acted_grid_sig == sig:
            return False               # 同一局面已出过手，等局面变化

        turn_en = SIDE_CN2EN[self.my_side]
        mv = self.engine.pick_move(grid_list, turn_en)
        if mv is None:
            self.log("决策 | 当前结果下无合规走法")
            self.acted_grid_sig = sig
            return False

        v = mv["verdict"]
        cap = f"，吃 {cell_name(*mv['captured'])}" if mv["captured"] else ""
        self.log(f"决策 | {v['label']}；{self.my_side}方走 {mv['name']}{cap}")

        executed = self.mouse.move_piece(mv["src"], mv["dst"], obs.region)

        if self.demo:
            # 只建议不执行：标记该局面已建议过，等真实局面变化
            self.acted_grid_sig = sig
            return True

        if not executed:
            return False
        self.acted_grid_sig = sig
        predicted = np.asarray(obs.grid, dtype=np.int8).copy()
        sr, sc = mv["src"]
        dr, dc = mv["dst"]
        piece = int(predicted[sr][sc])
        predicted[sr][sc] = EMPTY
        if mv["captured"]:
            predicted[mv["captured"][0]][mv["captured"][1]] = EMPTY
        predicted[dr][dc] = piece
        self.after_my_move = predicted
        self.awaiting = True
        self.await_since = time.perf_counter()
        self.inferred_turn = "羊" if self.my_side == "狼" else "狼"
        self.opponent_grid_sig = predicted.tobytes()
        predicted_count = self._record_position(
            predicted, SIDE_CN2EN[self.inferred_turn])
        self.half_moves += 1
        if predicted_count >= 5:
            self._terminal_sig = predicted.tobytes()
            self.log("终局 | 和棋（重复局面达到 5 次）")
            self._clear_game()
            self._last_initial = None
            return True
        if self.max_half_moves and self.half_moves >= self.max_half_moves:
            self._terminal_sig = predicted.tobytes()
            self.log(f"终局 | 和棋（达到 {self.max_half_moves} 步上限）")
            self._clear_game()
            self._last_initial = None
            return True
        time.sleep(self.post_move_delay)
        return True

    # ------------------------------------------------------------------
    def run(self, duration: float | None = None, should_stop=None):
        """运行自动下棋主循环。

        duration: 运行秒数（None=一直运行）；
        should_stop: 可选回调，返回 True 时优雅停止（交互菜单的 q/ESC 热键）。
        """
        eng = self.engine
        self.log(f"启动 | 阵营={self.side_arg} 演示={self.demo} 表库 max_k={eng.max_k}")
        t0 = time.perf_counter()
        last_status, last_sig = 0.0, None
        try:
            while duration is None or time.perf_counter() - t0 < duration:
                if should_stop is not None and should_stop():
                    self.log("停止 | 正在结束自动对局")
                    break
                try:
                    obs = self.observer.observe(interval=0.05)
                except RuntimeError as e:
                    self.log(f"检测 | {e}")
                    if should_stop is not None:
                        for _ in range(8):     # 等待期间仍可响应热键
                            if should_stop():
                                self.log("停止 | 正在结束自动对局")
                                return
                            time.sleep(0.25)
                    else:
                        time.sleep(2.0)
                    continue

                sig = obs.grid.tobytes()
                if is_initial_grid(obs.grid):
                    if self.half_moves > 0 or self.games == 0 or \
                            not np.array_equal(obs.grid, self._last_initial):
                        self._last_initial = obs.grid.copy()
                        self._start_new_game("检测到初始局面")
                        self._resolve_side(obs)

                if sig != last_sig:
                    last_sig = sig
                    if self.on_observation is not None:
                        self.on_observation(obs)
                    else:
                        self.log(self._format_observation(obs))

                try:
                    self.decide_and_act(obs)
                except KeyboardInterrupt as e:
                    if str(e):
                        self.log(f"停止 | {e}")
                        return
                    raise

                now = time.perf_counter()
                if now - last_status > 10.0:
                    last_status = now
                    self.log(f"状态 | 狼 {obs.wolves} 羊 {obs.sheep} "
                             f"回合 {self.inferred_turn or obs.turn or '未知'}")
                time.sleep(0.15)
        except KeyboardInterrupt:
            self.log("停止 | 用户中断")
        finally:
            self.log(f"结束 | 共识别 {self.games} 局")

    @staticmethod
    def _key_of(grid: np.ndarray, turn_en: str):
        """观测矩阵 -> rules 局面 key。"""
        from .engine import pos_key_of, game_from_grid
        g = game_from_grid([[int(v) for v in row] for row in grid], turn_en)
        return pos_key_of(g)

    def _record_position(self, grid: np.ndarray, turn_en: str | None) -> int:
        """记录稳定局面；连续轮询同一帧只计一次。"""
        if turn_en is None:
            return 0
        key = self._key_of(np.asarray(grid), turn_en)
        if key == self._last_position_key:
            return self.position_counts.get(key, 0)
        self._last_position_key = key
        count = self.position_counts.get(key, 0) + 1
        self.position_counts[key] = count
        return count

    @staticmethod
    def _format_observation(obs):
        lines = ["    A   B   C   D   E", "  ┌───┬───┬───┬───┬───┐"]
        for r in range(5):
            cells = " │ ".join(SYM[int(v)] for v in obs.grid[r])
            lines.append(f"{r + 1} │ {cells} │")
            if r < 4:
                lines.append("  ├───┼───┼───┼───┼───┤")
        lines.append("  └───┴───┴───┴───┴───┘")
        info = []
        if obs.faction:
            info.append(f"我方:{obs.faction}")
        if obs.turn:
            info.append(f"轮到:{obs.turn}")
        info.append(f"W:{obs.wolves} S:{obs.sheep}")
        return "\n".join(lines) + "\n  " + "  ".join(info)
