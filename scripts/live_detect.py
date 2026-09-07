#!/usr/bin/env python3
"""狼吃羊实时屏幕检测（轻量版）：整窗捕获 -> 标注图网格直套 -> 逐帧棋子检测。

流程:
1. osascript 找到小程序窗口，整窗抓取;
2. 启动时解析标注图的绿点，得到 5x5 归一化网格;
3. 根据棋局面判定阵营（3狼在最下排=狼方，180度旋转=羊方）;
4. 检测紫色边框区域判定轮到谁走棋;
5. 3 帧投票 + 2 帧确认即产出走子事件。

用法:
    python3 scripts/live_detect.py                 # 自动找游戏窗口
    python3 scripts/live_detect.py --duration 60   # 指定秒数
    python3 scripts/live_detect.py --region "2012,162,1012,1802"  # 手动指定窗口
预览窗口按 q 退出。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/PingFang.ttc"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wolves_cv.analyzer import GameAnalyzer
from wolves_cv.pieces import EMPTY, SHEEP, WOLF, PieceClassifier, CHAR

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "output_live"
ANNOTATION = HERE / "materials" / "棋盘网格线标注_红线绿点黑背景.png"

COLORS = {SHEEP: (255, 200, 60), WOLF: (80, 80, 235)}
VOTE_FRAMES = 3
CONFIRM_FRAMES = 2
RELOCATE_SECONDS = 3.0
WIN_KEYWORDS = ("狼吃羊", "小程序", "微信", "WeChat", "棋")
WIN_ASPECT = 1332 / 2479


def put_text_pil(img: np.ndarray, text: str, pos: tuple[int, int],
                 font_size: int = 20, color: tuple = (255, 255, 255),
                 bg_color: tuple = (0, 0, 0), draw_bg: bool = False) -> np.ndarray:
    """使用 PIL 绘制中文文字到 OpenCV 图像上。"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox(pos, text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if draw_bg:
        padding = 4
        bg_rect = [pos[0] - padding, pos[1] - padding,
                   pos[0] + text_w + padding, pos[1] + text_h + padding]
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(bg_rect, fill=(bg_color[0], bg_color[1], bg_color[2], 180))
        pil_img = pil_img.convert('RGBA')
        pil_img = Image.alpha_composite(pil_img, overlay)
        pil_img = pil_img.convert('RGB')
        draw = ImageDraw.Draw(pil_img)

    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def parse_annotation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """从标注图绿点解析 5x5 归一化网格坐标。"""
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


def parse_turn_region(path: Path) -> dict | None:
    """从标注图解析上方紫色边框区域（用于判断轮到谁走）。"""
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
        regions.append({
            "x": x / w, "y": y / h,
            "w": ww / w, "h": hh / h,
            "cy": cent[i][1] / h,
        })
    if not regions:
        return None
    regions.sort(key=lambda r: r["cy"])
    return regions[0]


def _osascript(script: str) -> str:
    return subprocess.run(["osascript", "-e", script],
                          capture_output=True, text=True).stdout.strip()


def find_game_window() -> tuple[int, int, int, int] | None:
    """返回小程序窗口完整矩形 (x, y, w, h)。"""
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
        print(f"[win] 小程序窗口: {names} {rect}")
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
    print(f"[win] 小程序窗口(枚举): {names} {rect}")
    return rect


class LiveSession:
    def __init__(self):
        self.pieces = PieceClassifier(HERE / "materials")
        self.nx, self.ny = parse_annotation(ANNOTATION)
        self.turn_region = parse_turn_region(ANNOTATION)
        self.history: deque[np.ndarray] = deque(maxlen=VOTE_FRAMES)
        self.committed: np.ndarray | None = None
        self.pending, self.pending_n = None, 0
        self.moves: list[dict] = []
        self.frame_idx = -1
        self.initial_grid = None
        self.game_initials: list[tuple[int, np.ndarray]] = []
        self.my_faction: str | None = None  # 我的阵营: "狼" 或 "羊"
        self.turn: str | None = None  # 当前轮到谁: "狼" 或 "羊"

    def grid_for(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        return self.nx * w, self.ny * h

    def classify_frame(self, frame_bgr: np.ndarray, vx, hy) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        spacing = float(vx[-1] - vx[0]) / 4
        grid, _ = self.pieces.classify(hsv, vx, hy, spacing)
        return grid

    def detect_my_faction(self, grid: np.ndarray) -> str | None:
        """根据初始棋局判定阵营。
        规则:
        - 3狼在最下排中央 + 15羊在上3排 = 狼方
        - 3狼在最上排中央 + 15羊在下3排 = 羊方（180度旋转）
        - 其他 = 不是新开局
        """
        wolves_count = (grid == WOLF).sum()
        sheep_count = (grid == SHEEP).sum()

        # 总棋子数应该是18（3狼+15羊）
        if wolves_count != 3 or sheep_count != 15:
            self.my_faction = None
            return None

        # 检查3狼在最下排中央（row 4, col 1,2,3）
        bottom_row = grid[4, :]
        wolves_bottom = (bottom_row[1:4] == WOLF).sum()
        if wolves_bottom == 3:
            # 检查羊是否都在上3排
            top_sheep = (grid[0:3, :] == SHEEP).sum()
            if top_sheep == 15:
                self.my_faction = "狼"
                print(f"[game] 检测到初始棋局: 狼方")
                return self.my_faction

        # 检查3狼在最上排中央（row 0, col 1,2,3）
        top_row = grid[0, :]
        wolves_top = (top_row[1:4] == WOLF).sum()
        if wolves_top == 3:
            # 检查羊是否都在下3排
            bottom_sheep = (grid[2:5, :] == SHEEP).sum()
            if bottom_sheep == 15:
                self.my_faction = "羊"
                print(f"[game] 检测到初始棋局: 羊方")
                return self.my_faction

        self.my_faction = None
        return None

    def detect_turn(self, frame_bgr: np.ndarray, win_w: int, win_h: int) -> str | None:
        """检测上方紫色区域判定轮到谁走棋。
        红色字体+黄色背景 = 轮到狼走
        白色字体+黄色背景 = 轮到羊走
        """
        if self.turn_region is None:
            return None

        reg = self.turn_region
        x1 = int(reg["x"] * win_w)
        y1 = int(reg["y"] * win_h)
        x2 = int((reg["x"] + reg["w"]) * win_w)
        y2 = int((reg["y"] + reg["h"]) * win_h)

        pad_x = max(10, int(0.02 * win_w))
        pad_y = max(10, int(0.02 * win_h))
        y1e = max(0, y1 - pad_y)
        y2e = min(frame_bgr.shape[0], y2 + pad_y)
        x1e = max(0, x1 - pad_x)
        x2e = min(frame_bgr.shape[1], x2 + pad_x)

        roi = frame_bgr[y1e:y2e, x1e:x2e]
        if roi.size == 0:
            return None

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]

        yellow_mask = (h > 20) & (h < 40) & (s > 100) & (v > 150)
        yellow_ratio = yellow_mask.sum() / max(yellow_mask.size, 1)

        if yellow_ratio < 0.1:
            return None

        red_mask = (((h < 15) | (h > 165)) & (s > 100) & (v > 100)) & yellow_mask
        red_ratio = red_mask.sum() / max(yellow_mask.sum(), 1)

        white_mask = (s < 50) & (v > 200) & yellow_mask
        white_ratio = white_mask.sum() / max(yellow_mask.sum(), 1)

        if red_ratio > 0.2:
            self.turn = "狼"
        elif white_ratio > 0.2:
            self.turn = "羊"
        else:
            avg_h = np.mean(h[yellow_mask]) if yellow_mask.any() else 0
            if avg_h < 15 or avg_h > 165:
                self.turn = "狼"
            else:
                self.turn = "羊"

        return self.turn

    @staticmethod
    def vote(history) -> np.ndarray:
        stack = np.stack(history)
        votes = np.stack([(stack == v).sum(0) for v in (EMPTY, SHEEP, WOLF)])
        return np.argmax(votes, 0).astype(np.int8)

    def update_events(self, stable: np.ndarray, t: float):
        if self.committed is None:
            if (stable != EMPTY).any():
                self.committed = stable.copy()
                self.initial_grid = stable.copy()
                self.game_initials.append((self.frame_idx, stable.copy()))
            return
        if np.array_equal(stable, self.committed):
            self.pending, self.pending_n = None, 0
            return
        if self.pending is not None and np.array_equal(stable, self.pending):
            self.pending_n += 1
        else:
            self.pending, self.pending_n = stable.copy(), 1
        if self.pending_n >= CONFIRM_FRAMES:
            ev = GameAnalyzer._diff_event(self.committed, stable)
            if ev is not None:
                ev.frame, ev.time_s = self.frame_idx, t
                self.moves.append(ev.to_dict())
                print(f"[move] {ev.to_dict()}")
                if ev.kind == "reset":
                    self.my_faction = None
                    self.game_initials.append((self.frame_idx, stable.copy()))
            self.committed = stable.copy()
            self.pending, self.pending_n = None, 0

    def annotate(self, frame: np.ndarray, vx, hy, stable: np.ndarray,
                 fps: float, win_w: int, win_h: int) -> np.ndarray:
        ann = frame.copy()

        # 绘制红色网格线
        for x in vx:
            cv2.line(ann, (int(x), int(hy[0])), (int(x), int(hy[-1])), (0, 0, 255), 2)
        for y in hy:
            cv2.line(ann, (int(vx[0]), int(y)), (int(vx[-1]), int(y)), (0, 0, 255), 2)

        # 绘制棋子
        spacing = float(vx[-1] - vx[0]) / 4
        for ri, y in enumerate(hy):
            for ci, x in enumerate(vx):
                v = stable[ri, ci]
                if v != EMPTY:
                    color = COLORS[v]
                    cv2.circle(ann, (int(x), int(y)), int(0.34 * spacing), color, 3)
                    ann = put_text_pil(ann, CHAR[v], (int(x) - 10, int(y) - 10),
                                       font_size=24, color=color)
                else:
                    cv2.circle(ann, (int(x), int(y)), 3, (0, 255, 0), -1)

        # 绘制右上角棋盘状态缩略图
        msz, mg, mo = 30, 4, 14
        ox = ann.shape[1] - mo - 5 * (msz + mg)
        oy = mo
        for ri in range(5):
            for ci in range(5):
                v = stable[ri, ci]
                c = {EMPTY: (60, 60, 60), SHEEP: (255, 200, 60), WOLF: (80, 80, 235)}[v]
                xx, yy = ox + ci * (msz + mg), oy + ri * (msz + mg)
                cv2.rectangle(ann, (xx, yy), (xx + msz, yy + msz), c, -1)

        # 绘制状态信息（左上角）
        info_lines = [
            f"F: {self.frame_idx}  FPS: {fps:.0f}",
            f"W: {int((stable == WOLF).sum())}  S: {int((stable == SHEEP).sum())}",
        ]
        if self.my_faction:
            info_lines.append(f"Mine: {self.my_faction}")
        if self.turn:
            info_lines.append(f"Turn: {self.turn}")

        font_size = 16
        line_height = 22
        x_pos = 10

        for i, line in enumerate(info_lines):
            y_pos = 10 + i * line_height
            ann = put_text_pil(ann, line, (x_pos, y_pos), font_size=font_size,
                               color=(255, 255, 255), bg_color=(0, 0, 0), draw_bg=True)
        return ann


def main():
    ap = argparse.ArgumentParser(description="狼吃羊实时屏幕检测（轻量版）")
    ap.add_argument("--duration", type=float, default=300, help="运行秒数")
    ap.add_argument("--region", type=str, default=None, help="手动指定窗口 x,y,w,h")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore", message=".*mss.*deprecated.*")
    import mss
    OUT_DIR.mkdir(exist_ok=True)
    sct = mss.mss()
    mon = sct.monitors[1]

    sess = LiveSession()
    if args.region:
        sess_region = tuple(int(v) for v in args.region.split(","))
        print(f"[win] 手动指定窗口: {sess_region}")
    else:
        sess_region = None

    default_w = 480
    default_h = int(default_w / WIN_ASPECT)
    cv2.namedWindow("wolves_live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wolves_live", default_w, default_h)

    vw = None
    states_f = open(OUT_DIR / "states.jsonl", "w", encoding="utf-8")
    t0 = time.perf_counter()
    last_relocate, last_status = 0.0, 0.0
    fps_ema = 0.0

    try:
        while time.perf_counter() - t0 < args.duration:
            ta = time.perf_counter()
            t_now = time.perf_counter() - t0

            if sess_region is None or t_now - last_relocate > RELOCATE_SECONDS:
                reg = find_game_window()
                last_relocate = time.perf_counter() - t0
                if reg is None:
                    if sess_region is None:
                        print("未找到小程序窗口, 等待... (或用 --region 手动指定)")
                        time.sleep(1.0)
                        continue
                    print(f"[win] 未找到窗口, 沿用上次区域 {sess_region}")
                else:
                    if sess_region is not None and reg != sess_region:
                        print(f"[win] 窗口变化: {sess_region} -> {reg}")
                    sess_region = reg
            x, y, w, h = sess_region

            frame = np.ascontiguousarray(
                np.array(sct.grab({"left": mon["left"] + x, "top": mon["top"] + y,
                                   "width": w, "height": h}))[:, :, :3])
            sess.frame_idx += 1

            vx, hy = sess.grid_for(w, h)
            raw = sess.classify_frame(frame, vx, hy)
            sess.history.append(raw)
            stable = sess.vote(sess.history)
            sess.update_events(stable, t_now)

            if sess.my_faction is None:
                sess.detect_my_faction(stable)

            if sess.frame_idx % 5 == 0:
                sess.detect_turn(frame, w, h)

            ann = sess.annotate(frame, vx, hy, stable, fps_ema, w, h)
            cv2.imshow("wolves_live", ann)

            if vw is None and sess.frame_idx >= 30:
                vh, vwc = ann.shape[0] & ~1, ann.shape[1] & ~1
                vw = cv2.VideoWriter(str(OUT_DIR / "annotated.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(10, round(fps_ema)), (vwc, vh))
            if vw is not None:
                vw.write(ann[:ann.shape[0] & ~1, :ann.shape[1] & ~1])

            states_f.write(json.dumps({
                "frame": sess.frame_idx, "t": round(t_now, 2),
                "grid": [[int(v) for v in row] for row in stable],
                "wolves": int((stable == WOLF).sum()),
                "sheep": int((stable == SHEEP).sum()),
                "my_faction": sess.my_faction,
                "turn": sess.turn,
            }, ensure_ascii=False) + "\n")

            tb = time.perf_counter()
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(tb - ta, 1e-6))
            if t_now - last_status > 5.0:
                last_status = t_now
                print(f"[stat] F:{sess.frame_idx}  FPS:{fps_ema:.0f}  "
                      f"W:{int((stable == WOLF).sum())} S:{int((stable == SHEEP).sum())}  "
                      f"Mine:{sess.my_faction} Turn:{sess.turn}")
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("用户退出")
                break
            states_f.flush()
    finally:
        if vw is not None:
            vw.release()
        states_f.close()
        cv2.destroyAllWindows()
        summary = {
            "frames": sess.frame_idx + 1,
            "duration_s": round(time.perf_counter() - t0, 1),
            "grid": {"nx": sess.nx.tolist(), "ny": sess.ny.tolist(),
                     "region": list(sess_region) if sess_region else None},
            "my_faction": sess.my_faction,
            "initial_grid": [[int(v) for v in row] for row in sess.initial_grid]
                            if sess.initial_grid is not None else None,
            "games": [{"start_frame": f0, "initial_grid": [[int(v) for v in row] for row in g]}
                      for f0, g in sess.game_initials],
            "moves": sess.moves,
        }
        with open(OUT_DIR / "moves.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"结束: {sess.frame_idx + 1} 帧, 事件 {len(sess.moves)} 条 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
