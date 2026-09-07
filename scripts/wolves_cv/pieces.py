"""棋子识别：交点磁盘区域的颜色判据 + 棋子素材模板(NCC)复核。

实测判据（视频与实时窗口两种渲染均验证）:
  狼 = 红色像素占比 > 0.25        (实测狼 0.33~0.78, 羊 <=0.14, 空/装饰 <=0.07)
  羊 = 黑色字纹占比 > 0.04 且非狼  (实测羊 0.07~0.09, 空/装饰 ~0.00)
  空 = 其余
棋盘左下/右下角的圆角装饰为无字纹木色块，天然被上述判据排除；
模板 NCC 作为状态变化时的复核（素材: materials/狼棋子.png 羊棋子.png）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

EMPTY, SHEEP, WOLF = 0, 1, 2
CHAR = {EMPTY: ".", SHEEP: "s", WOLF: "W"}

# 颜色判据（HSV, OpenCV 尺度 H:0-180 S/V:0-255）
RED_H0, RED_H1, RED_S, RED_V = 12, 165, 100, 80     # 狼盘: H<12 或 H>165
BLACK_V, BLACK_S = 90, 110                           # 羊的黑字纹
WOLF_TH = 0.25    # red_frac 超过 => 狼
SHEEP_TH = 0.04   # black_frac 超过 => 羊
NCC_CONFIRM = 0.5  # 状态变化时羊模板 NCC 需高于此值
DISK_R = 0.35     # 磁盘半径 = DISK_R * 网格间距


@dataclass
class DiskFeatures:
    red: float
    black: float


class PieceClassifier:
    """把一帧图 + 棋盘网格映射为 5x5 状态矩阵。"""

    def __init__(self, sprite_dir: str | Path | None = None):
        self._sprites: dict[str, np.ndarray] = {}
        self._sprite_D = 0
        if sprite_dir is not None:
            self._load_sprites(sprite_dir)

    # ------------------------------------------------------------------
    def _load_sprites(self, sprite_dir: Path | str) -> None:
        d = Path(sprite_dir)
        for key, name in (("wolf", "狼棋子.png"), ("sheep", "羊棋子.png")):
            p = d / name
            if p.exists():
                self._sprites[key] = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    def _template(self, key: str, D: int) -> np.ndarray | None:
        """按直径 D 缓存生成模板（透明角填充为羊皮纸色）。"""
        if key not in self._sprites:
            return None
        if self._sprite_D != D or key not in getattr(self, "_tpl_cache", {}):
            s = cv2.resize(self._sprites[key], (D, D), interpolation=cv2.INTER_AREA)
            alpha = (s[:, :, 3] > 128).astype(np.float32)
            fill = np.array([105, 163, 185], np.float32)
            rgb = s[:, :, :3].astype(np.float32) * alpha[:, :, None] + fill * (1 - alpha[:, :, None])
            if not hasattr(self, "_tpl_cache"):
                self._tpl_cache = {}
            self._tpl_cache[key] = rgb
            self._sprite_D = D
        return self._tpl_cache[key]

    # ------------------------------------------------------------------
    def verify(self, frame_bgr: np.ndarray, x: float, y: float, spacing: float):
        """模板 NCC 复核，返回 (wolf_ncc, sheep_ncc)。无素材时返回 (None, None)。"""
        D = int(round(0.72 * spacing))
        wt, st = self._template("wolf", D), self._template("sheep", D)
        if wt is None or st is None:
            return None, None
        pad = D // 2 + 8
        xi, yi = int(round(x)), int(round(y))
        roi = frame_bgr[max(0, yi - pad):yi + pad, max(0, xi - pad):xi + pad].astype(np.float32)
        if roi.shape[0] < D or roi.shape[1] < D:
            return -1.0, -1.0
        nw = float(cv2.matchTemplate(roi, wt, cv2.TM_CCOEFF_NORMED).max())
        ns = float(cv2.matchTemplate(roi, st, cv2.TM_CCOEFF_NORMED).max())
        return nw, ns

    # ------------------------------------------------------------------
    def classify(self, hsv: np.ndarray, vx: np.ndarray, hy: np.ndarray, spacing: float):
        """返回 (5x5 状态矩阵, 5x5 特征)。状态: 0空 1羊 2狼。"""
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        r = max(4, int(round(DISK_R * spacing)))
        dy, dx = self._disk_offsets(r)
        h, w = hsv.shape[:2]
        grid = np.zeros((len(hy), len(vx)), np.int8)
        feats = [[DiskFeatures(0.0, 0.0)] * len(vx) for _ in range(len(hy))]
        for ri, y in enumerate(hy):
            for ci, x in enumerate(vx):
                ys = np.clip(dy + int(round(y)), 0, h - 1)
                xs = np.clip(dx + int(round(x)), 0, w - 1)
                hh, ss, vv = H[ys, xs], S[ys, xs], V[ys, xs]
                n = max(len(hh), 1)
                red = float((((hh < RED_H0) | (hh > RED_H1)) & (ss >= RED_S) & (vv >= RED_V)).sum()) / n
                black = float(((vv <= BLACK_V) & (ss <= BLACK_S)).sum()) / n
                feats[ri][ci] = DiskFeatures(red, black)
                if red > WOLF_TH:
                    grid[ri, ci] = WOLF
                elif black > SHEEP_TH:
                    grid[ri, ci] = SHEEP
        return grid, feats

    def _disk_offsets(self, r: float):
        key = int(round(r))
        cache = getattr(self, "_mask_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        yy, xx = np.mgrid[-key:key + 1, -key:key + 1]
        inside = (yy ** 2 + xx ** 2) <= key * key
        dy, dx = yy[inside], xx[inside]
        self._mask_cache = (key, dy, dx)
        return dy, dx

    # ------------------------------------------------------------------
    @staticmethod
    def grid_str(grid: np.ndarray) -> str:
        return "/".join("".join(CHAR[v] for v in row) for row in grid)
