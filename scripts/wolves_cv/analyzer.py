"""对局分析器：逐帧棋盘+棋子状态 -> 时间平滑 -> 走子事件流。"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .board import BoardTracker
from .pieces import EMPTY, SHEEP, WOLF, PieceClassifier


@dataclass
class MoveEvent:
    frame: int
    time_s: float
    mover: str            # "wolf" | "sheep"
    kind: str             # "move" | "capture" | "place" | "complex"
    src: tuple | None     # (row, col)
    dst: tuple | None
    captured: list = field(default_factory=list)  # 被吃掉的子 [(row, col), ...]

    def to_dict(self):
        return {
            "frame": self.frame, "time_s": round(self.time_s, 2), "mover": self.mover,
            "kind": self.kind,
            "src": list(self.src) if self.src else None,
            "dst": list(self.dst) if self.dst else None,
            "captured": [list(c) for c in self.captured],
        }


@dataclass
class FrameResult:
    frame: int
    board_found: bool
    raw: np.ndarray            # 本帧 5x5
    stable: np.ndarray         # 时间平滑后 5x5
    wolves: int
    sheep: int
    ms: float                   # 单帧处理耗时


class GameAnalyzer:
    """流式分析器：喂入 BGR 帧即可，适用于实时与离线。"""

    def __init__(self, vote_window: int = 5, commit_stable: int = 3, fps: float = 15.0,
                 sprite_dir: str | None = None):
        if sprite_dir is None:
            here = Path(__file__).resolve()
            sprite_dir = here.parent.parent.parent / "materials"
        self.board = BoardTracker(piece_validator=self._board_has_pieces)
        self.pieces = PieceClassifier(sprite_dir)
        self.fps = fps
        self._history: deque[np.ndarray] = deque(maxlen=vote_window)
        self._committed: np.ndarray | None = None   # 已确认的稳定局面
        self._pending: np.ndarray | None = None
        self._pending_count = 0
        self.frame_idx = -1
        self.moves: list[MoveEvent] = []
        self.initial_state: np.ndarray | None = None
        self.game_initials: list[tuple[int, np.ndarray]] = []  # (帧号, 初始局面)
        self.commit_stable = commit_stable

    # ------------------------------------------------------------------
    def _board_has_pieces(self, frame_bgr: np.ndarray, vx: np.ndarray, hy: np.ndarray) -> bool:
        """棋盘锁定校验：真实对局棋盘的交点上应能识别出棋子。

        用于排除结算弹窗里的小棋盘图（其上棋子极小、颜色判据失效）。
        """
        spacing = (vx[-1] - vx[0] + hy[-1] - hy[0]) / (2 * (len(vx) - 1))
        if spacing < 40:  # 弹窗内小图
            return False
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        grid, _ = self.pieces.classify(hsv, vx, hy, spacing)
        return int((grid != EMPTY).sum()) >= 2

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        self.frame_idx += 1
        if not self.board.track(frame_bgr):
            # 丢失棋盘：沿用上次稳定局面，不产生事件
            raw = self._history[-1].copy() if self._history else np.zeros((5, 5), np.int8)
        else:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            raw, _ = self.pieces.classify(hsv, self.board.vx, self.board.hy, self.board.spacing)
        self._history.append(raw)
        stable = self._vote()
        self._update_events(stable)
        ms = (time.perf_counter() - t0) * 1000.0
        return FrameResult(
            frame=self.frame_idx, board_found=bool(self.board.found), raw=raw, stable=stable,
            wolves=int((stable == WOLF).sum()), sheep=int((stable == SHEEP).sum()), ms=ms,
        )

    # ------------------------------------------------------------------
    def _vote(self) -> np.ndarray:
        """滑动窗口逐格多数投票，抑制动画/高亮造成的瞬时误读。"""
        stack = np.stack(self._history)
        votes = np.stack([(stack == v).sum(0) for v in (EMPTY, SHEEP, WOLF)])  # 3x5x5
        return np.argmax(votes, axis=0).astype(np.int8)

    def _update_events(self, stable: np.ndarray) -> None:
        if self._committed is None:
            if (stable != EMPTY).any():
                self._committed = stable.copy()
                self.initial_state = stable.copy()
                self.game_initials.append((self.frame_idx, stable.copy()))
            return
        if np.array_equal(stable, self._committed):
            self._pending, self._pending_count = None, 0
            return
        # 局面变化需连续 commit_stable 帧一致才确认，避免动画中间态
        if self._pending is not None and np.array_equal(stable, self._pending):
            self._pending_count += 1
        else:
            self._pending, self._pending_count = stable.copy(), 1
        if self._pending_count >= self.commit_stable:
            event = self._diff_event(self._committed, stable)
            if event is not None:
                event.frame = self.frame_idx
                event.time_s = self.frame_idx / self.fps
                self.moves.append(event)
                if event.kind == "reset":
                    # 新对局：记录新初始局面
                    self.game_initials.append((self.frame_idx, stable.copy()))
            self._committed = stable.copy()
            self._pending, self._pending_count = None, 0

    # ------------------------------------------------------------------
    @staticmethod
    def _diff_event(old: np.ndarray, new: np.ndarray) -> MoveEvent | None:
        gone, appeared = [], []
        for r in range(old.shape[0]):
            for c in range(old.shape[1]):
                if old[r, c] != new[r, c]:
                    if old[r, c] != EMPTY:
                        gone.append((int(old[r, c]), (r, c)))
                    if new[r, c] != EMPTY:
                        appeared.append((int(new[r, c]), (r, c)))
        if not gone and not appeared:
            return None
        # 大规模同时变化 => 发牌/重开对局
        if len(gone) + len(appeared) >= 6:
            return MoveEvent(0, 0.0, "system", "reset", None, None)
        wolves_gone = [p for v, p in gone if v == WOLF]
        wolves_app = [p for v, p in appeared if v == WOLF]
        sheep_gone = [p for v, p in gone if v == SHEEP]
        sheep_app = [p for v, p in appeared if v == SHEEP]

        if len(gone) == 1 and len(appeared) == 1 and gone[0][0] == appeared[0][0]:
            v, sp = gone[0]
            return MoveEvent(0, 0.0, "wolf" if v == WOLF else "sheep", "move", sp, appeared[0][1])
        if not gone and sheep_app:  # 落子阶段（羊从手中上盘）
            return MoveEvent(0, 0.0, "sheep", "place", None, sheep_app[0])
        if len(wolves_gone) == 1 and len(wolves_app) == 1 and sheep_gone:
            return MoveEvent(0, 0.0, "wolf", "capture", wolves_gone[0], wolves_app[0], sheep_gone)
        if sheep_gone and not sheep_app and not wolves_app and not wolves_gone:
            return MoveEvent(0, 0.0, "wolf", "capture", None, None, sheep_gone)
        return MoveEvent(0, 0.0, "unknown", "complex", None, None, [])
