"""屏幕观测：定位游戏窗口 -> 抓帧 -> 5x5 棋子识别（投票稳定）->
阵营判定 + 走棋方检测。

算法核心复用 scripts/wolves_cv（棋盘网格/棋子判据），本模块只负责
窗口发现、抓帧调度与高层语义（回合/阵营/稳定局面）。

用法：
    obs = ScreenObserver()          # 自动找窗口
    o = obs.observe()               # Observation(grid, turn, faction, stable)
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent / "scripts"
for p in (str(_SCRIPTS_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

from wolves_cv.pieces import EMPTY, SHEEP, WOLF, PieceClassifier  # noqa: E402

ROOT = _HERE.parent
ANNOTATION = ROOT / "materials" / "棋盘网格线标注_红线绿点黑背景.png"

WIN_KEYWORDS = ("狼吃羊", "小程序", "微信", "WeChat", "棋")
WIN_ASPECT = 1332 / 2479
VOTE_FRAMES = 3
TURN_EVERY = 3


# ============================================================
# 窗口定位（macOS osascript）
# ============================================================

def _osascript(script: str) -> str:
    return subprocess.run(["osascript", "-e", script],
                          capture_output=True, text=True).stdout.strip()


def find_game_window(logger=print) -> tuple[int, int, int, int] | None:
    """返回小程序窗口完整矩形 (x, y, w, h)；找不到返回 None。"""
    fast = _osascript('''
    tell application "System Events"
        repeat with pname in {"WeChat", "微信"}
            try
                repeat with w in (windows of process pname whose name contains "狼吃羊")
                    set {px, py} to position of w
                    set {ww, hh} to size of w
                    return pname & "|" & px & "," & py & "," & ww & "," & hh
                end repeat
            end try
        end repeat
    end tell
    return ""
    ''')
    if fast:
        names, geo = fast.rsplit("|", 1)
        rect = tuple(int(v) for v in geo.split(","))
        logger(f"窗口 | {names} {rect}")
        return rect

    script = """
    tell application "System Events"
        set out to ""
        repeat with p in (every process whose background only is false)
            try
                repeat with w in windows of p
                    set {px, py} to position of w
                    set {ww, hh} to size of w
                    set out to out & (name of p) & "/" & (name of w) & "|" & px & "," & py & "," & ww & "," & hh & linefeed
                end repeat
            end try
        end repeat
        return out
    end tell
    """
    candidates = []
    for line in _osascript(script).splitlines():
        try:
            names, geo = line.rsplit("|", 1)
            x, y, w, h = (int(v) for v in geo.split(","))
        except ValueError:
            continue
        if w < 300 or h < 500 or not 0.35 <= w / h <= 0.75:
            continue
        named = any(k in names for k in WIN_KEYWORDS)
        aspect_err = abs(w / h - WIN_ASPECT) / WIN_ASPECT
        candidates.append((not named, aspect_err, -(w * h), (x, y, w, h), names))
    if not candidates:
        return None
    candidates.sort()
    _, _, _, rect, names = candidates[0]
    logger(f"窗口 | {names} {rect}")
    return rect


# ============================================================
# 标注图解析（网格归一化坐标 + 回合指示区域）
# ============================================================

def parse_annotation(path: Path = ANNOTATION) -> tuple[np.ndarray, np.ndarray]:
    """从标注图绿点解析 5x5 归一化网格坐标 (nx, ny)。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"标注图不存在: {path}")
    bgr = img[:, :, :3]
    h, w = bgr.shape[:2]
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    mask = ((g > 200) & (r < 120) & (b < 120)).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    pts = [cent[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]

    def cluster(vals: list[float]) -> list[float]:
        vals = sorted(vals)
        groups: list[list[float]] = [[vals[0]]]
        for v in vals[1:]:
            if v - groups[-1][-1] <= 0.02 * max(h, w):
                groups[-1].append(v)
            else:
                groups.append([v])
        return [float(np.mean(gp)) for gp in groups]

    xs, ys = cluster([p[0] for p in pts]), cluster([p[1] for p in pts])
    if len(xs) != 5 or len(ys) != 5:
        raise RuntimeError(f"标注图解析到 {len(xs)}x{len(ys)} 网格, 期望 5x5")
    return np.array(xs) / w, np.array(ys) / h


def parse_turn_region(path: Path = ANNOTATION) -> dict | None:
    """从标注图解析上方紫色边框区域（判断轮到谁走）。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    bgr = img[:, :, :3]
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 150) &
            (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 30)).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    regions = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 100:
            continue
        regions.append({"x": x / w, "y": y / h, "w": ww / w, "h": hh / h,
                        "cy": cent[i][1] / h})
    if not regions:
        return None
    regions.sort(key=lambda r: r["cy"])
    return regions[0]


# ============================================================
# 高层观测
# ============================================================

@dataclass
class Observation:
    grid: np.ndarray                 # 5x5 int8：0空 1羊 2狼
    turn: str | None = None          # "狼" | "羊" | None（未检出）
    faction: str | None = None       # 我方阵营（仅初始局面可自动判定）
    wolves: int = 0
    sheep: int = 0
    region: tuple | None = None      # 窗口矩形
    frames: int = 0                  # 参与投票的帧数

    @property
    def is_initial(self) -> bool:
        """是否标准初始局面（任一方向）。"""
        g = self.grid
        if (g[4, 1:4] == WOLF).all() and (g[0:3, :] == SHEEP).all():
            return True
        return bool((g[0, 1:4] == WOLF).all() and (g[2:5, :] == SHEEP).all())

    def board_ascii(self) -> str:
        sym = {EMPTY: ".", SHEEP: "s", WOLF: "W"}
        return "\n".join("".join(sym[int(v)] for v in row) for row in self.grid)


class ScreenObserver:
    """逐帧观测游戏窗口，产出稳定的 5x5 局面 + 回合/阵营。"""

    def __init__(self, region: tuple[int, int, int, int] | None = None,
                 sprites_dir: Path | None = None,
                 relocate_seconds: float = 60.0, logger=print):
        import warnings
        warnings.filterwarnings("ignore", message=".*mss.*deprecated.*")
        self.sct = None
        self.mon = None
        self.region = region
        self.fixed_region = region is not None
        self.relocate_seconds = relocate_seconds
        self.log = logger
        self._last_relocate = 0.0
        self.nx, self.ny = parse_annotation()
        self.turn_region = parse_turn_region()
        self.classifier = PieceClassifier(sprites_dir or ROOT / "materials")
        self.history: list[np.ndarray] = []
        self.turn: str | None = None
        self.faction: str | None = None
        self.turn_frame = -10 ** 9
        self.frame_idx = -1

    # ---- 基础 ----
    def ensure_capture(self):
        if self.sct is None:
            import mss
            self.sct = mss.mss()
            self.mon = self.sct.monitors[1]

    def locate(self) -> tuple[int, int, int, int]:
        """返回当前窗口区域；自动模式下定期重定位。"""
        now = time.perf_counter()
        if self.region is not None and (self.fixed_region or
                                        now - self._last_relocate < self.relocate_seconds):
            return self.region
        self._last_relocate = now
        rect = find_game_window(self.log)
        if rect is not None:
            if self.region is not None and rect != self.region:
                self.log(f"窗口 | 区域变化 {self.region} -> {rect}")
            self.region = rect
            return rect
        if self.region is None:
            raise RuntimeError("未找到小程序窗口，请打开游戏或用 --region x,y,w,h 指定")
        self.log(f"窗口 | 未重新定位，沿用 {self.region}")
        return self.region

    def grab(self, region: tuple[int, int, int, int]) -> np.ndarray:
        self.ensure_capture()
        x, y, w, h = region
        frame = np.ascontiguousarray(
            np.array(self.sct.grab({"left": self.mon["left"] + x,
                                    "top": self.mon["top"] + y,
                                    "width": w, "height": h}))[:, :, :3])
        return frame

    def grid_for(self, w: int, h: int):
        return self.nx * w, self.ny * h

    # ---- 单帧识别 ----
    def classify_frame(self, frame_bgr: np.ndarray, vx, hy) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        spacing = float(vx[-1] - vx[0]) / 4
        grid, _ = self.classifier.classify(hsv, vx, hy, spacing)
        return grid

    @staticmethod
    def vote(history) -> np.ndarray:
        stack = np.stack(history)
        votes = np.stack([(stack == v).sum(0) for v in (EMPTY, SHEEP, WOLF)])
        return np.argmax(votes, 0).astype(np.int8)

    def detect_turn(self, frame_bgr: np.ndarray, win_w: int, win_h: int) -> str | None:
        """红色字体=狼走，白色字体=羊走（黄色背景的紫色提示区）。"""
        if self.turn_region is None:
            return None
        reg = self.turn_region
        x1, y1 = int(reg["x"] * win_w), int(reg["y"] * win_h)
        x2 = int((reg["x"] + reg["w"]) * win_w)
        y2 = int((reg["y"] + reg["h"]) * win_h)
        pad_x = max(10, int(0.02 * win_w))
        pad_y = max(10, int(0.02 * win_h))
        roi = frame_bgr[max(0, y1 - pad_y):min(frame_bgr.shape[0], y2 + pad_y),
                        max(0, x1 - pad_x):min(frame_bgr.shape[1], x2 + pad_x)]
        if roi.size == 0:
            return None
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]
        yellow_mask = (h > 20) & (h < 40) & (s > 100) & (v > 150)
        if yellow_mask.sum() / max(yellow_mask.size, 1) < 0.1:
            return self.turn
        red_ratio = ((((h < 15) | (h > 165)) & (s > 100) & (v > 100)) & yellow_mask
                     ).sum() / max(yellow_mask.sum(), 1)
        white_ratio = ((s < 50) & (v > 200) & yellow_mask).sum() / max(yellow_mask.sum(), 1)
        if red_ratio > 0.2:
            self.turn = "狼"
        elif white_ratio > 0.2:
            self.turn = "羊"
        else:
            avg_h = np.mean(h[yellow_mask]) if yellow_mask.any() else 0
            self.turn = "狼" if (avg_h < 15 or avg_h > 165) else "羊"
        return self.turn

    def detect_faction(self, grid: np.ndarray) -> str | None:
        """根据初始布局自动判定我方阵营（3狼靠下=狼方；180°旋转=羊方）。"""
        if (grid == WOLF).sum() != 3 or (grid == SHEEP).sum() != 15:
            return None
        if (grid[4, 1:4] == WOLF).all() and (grid[0:3, :] == SHEEP).all():
            self.faction = "狼"
        elif (grid[0, 1:4] == WOLF).all() and (grid[2:5, :] == SHEEP).all():
            self.faction = "羊"
        else:
            return None
        return self.faction

    # ---- 观测入口 ----
    def observe(self, vote_frames: int = VOTE_FRAMES,
                interval: float = 0.12) -> Observation:
        """连续抓 vote_frames 帧 → 多数投票 → 稳定局面 + 回合检测。"""
        region = self.locate()
        x, y, w, h = region
        vx, hy = self.grid_for(w, h)
        history: list[np.ndarray] = []
        turn = None
        for i in range(vote_frames):
            frame = self.grab(region)
            self.frame_idx += 1
            history.append(self.classify_frame(frame, vx, hy))
            if i % TURN_EVERY == 0:
                turn = self.detect_turn(frame, w, h) or turn
            if i < vote_frames - 1:
                time.sleep(interval)
        grid = self.vote(history)
        faction = None
        if self.faction is None:
            faction = self.detect_faction(grid)
        else:
            faction = self.faction
        return Observation(
            grid=grid, turn=self.turn, faction=faction,
            wolves=int((grid == WOLF).sum()), sheep=int((grid == SHEEP).sum()),
            region=tuple(region), frames=len(history),
        )

    def wait_stable_grid(self, confirmations: int = 2, interval: float = 0.25,
                         timeout: float = 20.0) -> Observation | None:
        """等待局面在连续多次观测中保持一致（动画结束后调用）。"""
        t0 = time.perf_counter()
        last = None
        hits = 0
        while time.perf_counter() - t0 < timeout:
            obs = self.observe(vote_frames=2, interval=interval / 2)
            if last is not None and np.array_equal(obs.grid, last):
                hits += 1
            else:
                hits, last = 1, obs.grid.copy()
            if hits >= confirmations:
                return obs
            time.sleep(interval)
        return None
