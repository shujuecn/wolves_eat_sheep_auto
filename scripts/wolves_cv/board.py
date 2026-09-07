"""棋盘定位：等间距五线"梳状"组合搜索 + 双线边框验证 + 多帧持久性锁定。

棋盘结构（由标注图确认）：5x5 可落子交点网格，四边为双层装饰边框
（两条平行线相距约 0.1 倍网格间距，中间为浅色缝隙）。中央 2x2 格带 X 斜线。
"""
from __future__ import annotations

import cv2
import numpy as np

GRID_N = 5  # 5 条竖线 x 5 条横线 => 25 个交点


def _line_mask(hsv: np.ndarray) -> np.ndarray:
    """网格线为深棕色（低亮度、中等饱和度），底色为浅羊皮纸。"""
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    return ((v < 150) & (s > 60)).astype(np.float32)


def _comb_candidates(profile: np.ndarray, d_min: int, d_max: int, n_lines: int = 4,
                     top_k: int = 10):
    """搜索 n_lines 条等间距线的候选组合，返回按得分排序的 [(score, offset, spacing), ...]。

    每个间距保留最优与次优两个互不重叠的偏移，保证较弱的正确组合也在候选中。
    """
    prof = cv2.blur(profile.reshape(1, -1), (5, 1)).ravel()  # 平滑容忍线条抖动
    n = len(prof)
    results = []
    for d in range(d_min, min(d_max, (n - 1) // (n_lines - 1)) + 1):
        m = n - (n_lines - 1) * d
        if m <= 0:
            break
        idx = np.arange(m)
        s = np.zeros(m, np.float32)
        for k in range(n_lines):
            s += prof[idx + k * d]
        for _ in range(2):  # 最优 + 次优偏移
            o = int(np.argmax(s))
            strengths = np.array([prof[o + k * d] for k in range(n_lines)])
            if strengths.min() >= 0.5 * strengths.mean():
                results.append((float(s[o]), o, d))
            lo, hi = max(o - d // 2, 0), min(o + d // 2 + 1, m)
            s[lo:hi] = -1.0
    results.sort(reverse=True)
    return results[:top_k]


def _grid_candidates(profile: np.ndarray, d_min: int, d_max: int) -> list[tuple[float, int, int]]:
    """生成 5 线网格候选：4 线组合 + 双线边框验证的端点外推。

    棋盘边线可能被 UI 削弱，但内部 4 条线强且稳定；由 4 线组合向两端
    外推第 5 条线，只有外推端通过双线边框验证的组合被保留。
    返回 [(score, offset, spacing)]，5 条线位于 offset + k*spacing, k=0..4。
    """
    out: list[tuple[float, int, int]] = []
    for score, o, d in _comb_candidates(profile, d_min, d_max, n_lines=4, top_k=24):
        for ext in (o - d, o + 3 * d + d):
            if ext < 0 or ext + 4 * d >= len(profile):
                continue
            direction = -1 if ext == o - d else +1
            if not _twin_border_ok(profile, ext, direction, d):
                continue  # 外推端不是双线边框
            far = o + 3 * d if direction < 0 else o
            far_dir = +1 if direction < 0 else -1
            if not _twin_border_ok(profile, far, far_dir, d):
                continue  # 另一端也必须是双线边框（排除内部 4 线组合）
            out.append((score + float(profile[ext]), ext if direction < 0 else o, d))
    out.sort(reverse=True)
    return out


def _twin_border_ok(prof: np.ndarray, pos: float, direction: int, spacing: float) -> bool:
    """棋盘边框为双线：在外侧 0.06~0.22 倍间距处应有一条平行线，且两线之间有浅色凹谷。

    区分: 内部网格线（无伴随线）、UI 粗条（虽宽但无凹谷）。
    """
    n = len(prof)
    lo = int(round(pos + direction * spacing * 0.06))
    hi = int(round(pos + direction * spacing * 0.22))
    if direction < 0:
        lo, hi = hi, lo
    lo, hi = max(lo, 0), min(hi, n - 1)
    if hi - lo < 2:
        return False
    band = prof[lo:hi + 1]
    p0 = prof[int(round(pos))]
    if band.max() < 0.5 * max(p0, 1e-6):
        return False
    t = lo + int(np.argmax(band))
    a, b = sorted((int(round(pos)), t))
    mid = prof[a:b + 1]
    return float(mid.min()) < 0.45 * min(prof[a], prof[b])


class BoardTracker:
    """逐帧棋盘定位器。

    found 仅在锁定并验证棋盘后为 True；
    弹窗遮挡等导致的弱帧沿用最近位置，连续丢失则重新搜索。
    """

    def __init__(self, search_scale: float = 0.5, smooth: float = 0.4,
                 persist_frames: int = 10, piece_validator=None):
        self.search_scale = search_scale
        self.smooth = smooth
        self.persist_frames = persist_frames          # 候选需连续出现多少次才锁定
        self.piece_validator = piece_validator        # callable(frame, vx, hy) -> bool
        self.vx: np.ndarray | None = None
        self.hy: np.ndarray | None = None
        self.found = False
        self.confidence = 0.0
        self._miss = 0
        self._cand: tuple[np.ndarray, np.ndarray] | None = None
        self._cand_hits = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _piece_blobs(frame_bgr: np.ndarray):
        """检测棋子圆盘（红狼/米羊），返回 (圆心x, 圆心y, 半径) 列表。"""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        masks = [
            (((h < 12) | (h > 165)) & (s > 100) & (v > 80)),          # 狼：红盘
            ((h >= 8) & (h <= 42) & (s >= 110) & (v >= 120)),          # 羊：米盘
        ]
        out = []
        minr, maxr = 0.015 * min(frame_bgr.shape[:2]), 0.16 * min(frame_bgr.shape[:2])
        min_area = 2.0 * np.pi * minr * minr  # 相对阈值, 兼容不同分辨率/棋盘大小
        for m in masks:
            mm = (m.astype(np.uint8)) * 255
            mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                (cx, cy), r = cv2.minEnclosingCircle(c)
                if a > min_area and minr < r < maxr and a / (np.pi * r * r) > 0.55:
                    out.append((cx, cy, r))
        if len(out) >= 4:
            # 半径中位数过滤: 剔除移动提示光斑(两列中点的大光圈)与 UI 小图标,
            # 真棋子半径高度一致
            med = float(np.median([b[2] for b in out]))
            out = [b for b in out if 0.7 * med <= b[2] <= 1.35 * med]
        return out

    @staticmethod
    def _cluster_1d(values, tol):
        """把一维坐标按间距 tol 聚类，返回各类中心。"""
        vals = sorted(values)
        clusters, cur = [], [vals[0]]
        for x in vals[1:]:
            if x - cur[-1] <= tol:
                cur.append(x)
            else:
                clusters.append(sum(cur) / len(cur))
                cur = [x]
        clusters.append(sum(cur) / len(cur))
        return clusters

    def _piece_anchored_search(self, frame_bgr: np.ndarray):
        """棋子锚定定位：棋子必在交点上，聚类棋子圆心反推 5x5 等距网格。

        网格间距用棋子半径约束（棋子直径≈0.7 格距）消除倍频歧义。
        """
        blobs = self._piece_blobs(frame_bgr)
        if len(blobs) < 4:
            return None
        med_r = float(np.median([b[2] for b in blobs]))
        sp_expect = 2.9 * med_r
        xs = self._cluster_1d([b[0] for b in blobs], 0.35 * sp_expect)
        ys = self._cluster_1d([b[1] for b in blobs], 0.35 * sp_expect)
        if len(xs) < 3 or len(ys) < 3:
            return None

        def fit_grid(clusters):
            diffs = [b - a for a, b in zip(clusters, clusters[1:])]
            if min(diffs) < 0.45 * sp_expect:
                return None
            # 候选间距: 每个差分可能是 1~5 步跨距; RANSAC 拟合容忍离群簇
            # (移动提示光斑等干扰), 只要求多数簇落在等距网格上
            cands = {d / j for d in diffs for j in range(1, 6)}
            limit = max(frame_bgr.shape[:2])
            need = max(3, int(np.ceil(0.55 * len(clusters))))
            best = None  # (inliers, err, grid)
            for sp in sorted(cands):
                if sp < 0.6 * sp_expect or sp > 1.7 * sp_expect:
                    continue
                lo = max(-sp, clusters[0] - sp)
                hi = min(limit - 4 * sp, clusters[-1])
                if hi < lo:
                    continue
                for o in np.arange(lo, hi + 1e-6, sp / 8):
                    model = o + np.arange(GRID_N) * sp
                    idx = [int(np.argmin(np.abs(model - c))) for c in clusters]
                    inl = [(m, c) for m, c in zip(idx, clusters)
                           if abs(model[m] - c) < 0.22 * sp]
                    if len({m for m, _ in inl}) < need:
                        continue
                    A = np.stack([np.array([m for m, _ in inl], float),
                                  np.ones(len(inl))], 1)
                    coef, *_ = np.linalg.lstsq(A, np.array([c for _, c in inl]), rcond=None)
                    grid = coef[0] * np.arange(GRID_N) + coef[1]
                    if grid[0] < -5 or grid[-1] > limit + 5:
                        continue
                    key = (len(inl), -abs(sp - sp_expect))
                    if best is None or key > best[0]:
                        best = (key, grid)
            return best[1] if best else None

        vx, hy = fit_grid(xs), fit_grid(ys)
        if vx is None or hy is None:
            return None
        span_x, span_y = vx[-1] - vx[0], hy[-1] - hy[0]
        if abs(span_x - span_y) > 0.12 * max(span_x, span_y):
            return None  # 方形棋盘约束
        return self._refine_grid(vx, hy, blobs)

    @staticmethod
    def _refine_grid(vx: np.ndarray, hy: np.ndarray, blobs):
        """用吸附到网格的棋子圆心做每轴最小二乘精修（平移+缩放）。"""
        sp = (vx[-1] - vx[0]) / 4
        for axis_idx, coords in ((0, vx), (1, hy)):
            snapped = []
            for b in blobs:
                c = b[axis_idx]
                k = int(np.argmin(np.abs(coords - c)))
                if abs(coords[k] - c) < 0.25 * sp:
                    snapped.append((k, c))
            if len(snapped) < 3 or len({k for k, _ in snapped}) < 2:
                continue
            idx = np.array([k for k, _ in snapped], float)
            val = np.array([c for _, c in snapped], float)
            A = np.stack([idx, np.ones(len(idx))], 1)
            coef, *_ = np.linalg.lstsq(A, val, rcond=None)
            model = coef[0] * np.arange(GRID_N) + coef[1]
            if np.abs(model - coords).max() < 0.4 * sp:
                if axis_idx == 0:
                    vx = model
                else:
                    hy = model
        return vx, hy

    # ------------------------------------------------------------------
    def _search(self, frame_bgr: np.ndarray):
        """全帧搜索，返回 (vx, hy)（原图尺度）或 None。

        竖/横各取前 K 个等距组合 -> 间距匹配配对 -> 双线边框验证 -> 棋子验证。
        """
        sc = self.search_scale
        small = cv2.resize(frame_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = _line_mask(hsv)
        colsum = mask.sum(0)
        rowsum = mask.sum(1)
        h, w = mask.shape
        d0, d1 = int(110 * sc), int(340 * sc)
        # 间距范围自适应窗口大小: 棋盘横向占 4 个间距, 需能放进画面
        w_px, h_px = mask.shape[1], mask.shape[0]
        d_top = int(min(w_px, h_px) / 4.2)
        d0 = max(int(0.22 * d_top), int(50 * sc))
        d1 = min(d1, d_top) if d_top > d0 else d0
        vcands = _grid_candidates(colsum, d0, d1)
        hcands = _grid_candidates(rowsum, d0, d1)
        if not vcands or not hcands:
            return None
        pairs = []
        for vs, ox, dv in vcands:
            for hs, oy, dh in hcands:
                if abs(dv - dh) > 0.08 * max(dv, dh):
                    continue  # 方形棋盘两组间距应一致
                pairs.append((vs + hs, ox, dv, oy, dh))
        pairs.sort(reverse=True)
        limit = max(h, w)
        for score, ox, dv, oy, dh in pairs:
            vx = (ox + np.arange(GRID_N) * dv) / sc
            hy = (oy + np.arange(GRID_N) * dh) / sc
            conf = min(1.0, score / (5 * 0.30 * limit))
            if conf < 0.2:
                continue
            if self.piece_validator is not None and not self.piece_validator(frame_bgr, vx, hy):
                continue
            self.confidence = float(conf)
            return vx, hy
        # 梳状搜索失败 => 棋子锚定兜底（适配边框渲染不同的窗口）
        res = self._piece_anchored_search(frame_bgr)
        if res is not None:
            vx, hy = res
            if self.piece_validator is None or self.piece_validator(frame_bgr, vx, hy):
                self.confidence = 0.8
                return vx, hy
        return None

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> bool:
        """搜索并尝试锁定棋盘（带多帧持久性验证）。"""
        res = self._search(frame_bgr)
        if res is None:
            self._cand_hits = max(0, self._cand_hits - 2)  # 搜索失败削弱候选
            if self._cand_hits == 0:
                self._cand = None
            if self.vx is not None:
                self.found = True  # 沿用旧位置
                return True
            self.found = False
            return False
        vx, hy = res
        close = False
        if self._cand is not None:
            close = (np.abs(self._cand[0] - vx).max() < 12 and
                     np.abs(self._cand[1] - hy).max() < 12)
        if close:
            self._cand_hits += 1
        else:
            self._cand, self._cand_hits = (vx, hy), 1
        if self._cand_hits >= self.persist_frames:
            self.vx, self.hy = self._cand
            self.found, self._miss = True, 0
            return True
        if self.vx is not None:
            self.found = True
            return True
        self.found = False
        return False

    # ------------------------------------------------------------------
    def track(self, frame_bgr: np.ndarray, tol: int = 8) -> bool:
        """每帧调用：局部复核线位并做等间距投影，防漂移。"""
        if self.vx is None or not self.found:
            return self.detect(frame_bgr)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = _line_mask(hsv)
        conf = 0.0
        for axis in (0, 1):
            old = self.vx if axis == 0 else self.hy
            profile = mask.sum(0) if axis == 0 else mask.sum(1)
            limit = mask.shape[1] if axis == 0 else mask.shape[0]
            new = old.copy()
            ok_lines = 0
            for k, x in enumerate(old):
                lo, hi = max(int(x - tol), 0), int(x + tol + 1)
                seg = profile[lo:hi]
                if len(seg) == 0 or seg.max() < limit * 0.08:
                    continue
                m = int(np.argmax(seg)) + lo
                y0 = profile[m - 1] if m >= 1 else profile[m]
                y2 = profile[m + 1] if m + 1 < len(profile) else profile[m]
                denom = y0 - 2 * profile[m] + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-6 else 0.0
                new[k] = m + float(np.clip(delta, -0.5, 0.5))
                ok_lines += 1
                conf = max(conf, float(seg.max()) / limit)
            if ok_lines >= 3:
                # 等间距投影：最小二乘拟合 index->position，拉回均匀网格
                idx = np.arange(GRID_N, dtype=np.float64)
                A = np.stack([idx, np.ones(GRID_N)], 1)
                coef, *_ = np.linalg.lstsq(A, new, rcond=None)
                model = A @ coef
                # 偏离模型的线不采纳（可能跟丢到了相邻 UI 线）
                blended = np.where(np.abs(new - model) < 0.25 * (model[-1] - model[0]) / (GRID_N - 1),
                                   new, model)
                updated = (1 - self.smooth) * old + self.smooth * blended
                if axis == 0:
                    self.vx = updated
                else:
                    self.hy = updated
        self.confidence = float(min(conf, 1.0))
        ok = self.confidence > 0.10
        self._miss = 0 if ok else self._miss + 1
        self.found = ok
        if self._miss > 20:
            self.vx = None  # 彻底丢失，重新搜索
            self.found = False
        return self.found

    # ------------------------------------------------------------------
    @property
    def spacing(self) -> float:
        """平均网格间距（像素）。"""
        if self.vx is None:
            return 0.0
        return float((self.vx[-1] - self.vx[0] + self.hy[-1] - self.hy[0]) / (2 * (GRID_N - 1)))

    @property
    def corners(self):
        """棋盘四角 (x0, y0, x1, y1)。"""
        return (float(self.vx[0]), float(self.hy[0]), float(self.vx[-1]), float(self.hy[-1]))
