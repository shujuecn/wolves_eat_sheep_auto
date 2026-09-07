#!/usr/bin/env python3
"""模拟鼠标移动棋子。

坐标系: 左上角 A1, 右下角 E5
移动方式: 先点目标棋子，再点目的地

用法:
    python3 scripts/move_piece.py C3 C4   # 从 C3 移动到 C4
    python3 scripts/move_piece.py --region "2012,162,1012,1802" A1 B2
    python3 scripts/move_piece.py --demo A1 B2  # 演示模式
"""
from __future__ import annotations

import argparse
import subprocess
import time

import pyautogui

pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

# 从标注图解析的网格归一化坐标
NX = [0.146, 0.323, 0.500, 0.676, 0.852]  # A-E 列
NY = [0.337, 0.429, 0.521, 0.613, 0.705]  # 1-5 行

COL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
ROW_MAP = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}


def _osascript(script: str) -> str:
    return subprocess.run(["osascript", "-e", script],
                          capture_output=True, text=True).stdout.strip()


def find_game_window() -> tuple[int, int, int, int] | None:
    """返回小程序窗口矩形 (x, y, w, h)。"""
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
        _, geo = fast.rsplit("|", 1)
        return tuple(int(v) for v in geo.split(","))
    return None


def parse_position(pos: str) -> tuple[int, int]:
    """解析棋盘坐标，如 'A1' -> (0, 0)。"""
    if len(pos) != 2:
        raise ValueError(f"无效坐标: {pos}")
    col_char = pos[0].upper()
    row_char = pos[1]
    if col_char not in COL_MAP:
        raise ValueError(f"无效列: {col_char}，应为 A-E")
    if row_char not in ROW_MAP:
        raise ValueError(f"无效行: {row_char}，应为 1-5")
    return ROW_MAP[row_char], COL_MAP[col_char]


def grid_to_pixel(row: int, col: int, win_x: int, win_y: int,
                  win_w: int, win_h: int) -> tuple[int, int]:
    """将 5x5 网格坐标转换为屏幕像素坐标。"""
    px = win_x + int(NX[col] * win_w)
    py = win_y + int(NY[row] * win_h)
    return px, py


def move_piece(src_row: int, src_col: int, dst_row: int, dst_col: int,
               win_x: int, win_y: int, win_w: int, win_h: int,
               demo: bool = False):
    """双击源棋子（首次聚焦窗口），再单击目的地。"""
    src_px, src_py = grid_to_pixel(src_row, src_col, win_x, win_y, win_w, win_h)
    dst_px, dst_py = grid_to_pixel(dst_row, dst_col, win_x, win_y, win_w, win_h)

    src_name = list(COL_MAP.keys())[src_col] + list(ROW_MAP.keys())[src_row]
    dst_name = list(COL_MAP.keys())[dst_col] + list(ROW_MAP.keys())[dst_row]

    print(f"移动: {src_name} -> {dst_name}")
    print(f"像素: ({src_px},{src_py}) -> ({dst_px},{dst_py})")

    if demo:
        print("[演示模式] 不执行实际移动")
        return

    # 双击源棋子（第一次聚焦窗口，第二次选中棋子）
    pyautogui.click(src_px, src_py)
    time.sleep(0.15)
    pyautogui.click(src_px, src_py)
    time.sleep(0.3)

    # 单击目的地
    pyautogui.click(dst_px, dst_py)
    time.sleep(0.2)

    print("移动完成")


def main():
    ap = argparse.ArgumentParser(description="模拟鼠标移动棋子")
    ap.add_argument("src", type=str, help="源位置 (如 C3)")
    ap.add_argument("dst", type=str, help="目标位置 (如 C4)")
    ap.add_argument("--region", type=str, default=None, help="手动指定窗口 x,y,w,h")
    ap.add_argument("--demo", action="store_true", help="演示模式，只显示坐标不移动")
    args = ap.parse_args()

    try:
        src_row, src_col = parse_position(args.src)
        dst_row, dst_col = parse_position(args.dst)
    except ValueError as e:
        print(f"错误: {e}")
        return

    if args.region:
        win_x, win_y, win_w, win_h = (int(v) for v in args.region.split(","))
        print(f"使用指定窗口: ({win_x},{win_y}) {win_w}x{win_h}")
    else:
        rect = find_game_window()
        if rect is None:
            print("错误: 未找到游戏窗口")
            return
        win_x, win_y, win_w, win_h = rect
        print(f"找到窗口: ({win_x},{win_y}) {win_w}x{win_h}")

    move_piece(src_row, src_col, dst_row, dst_col, win_x, win_y, win_w, win_h,
               demo=args.demo)


if __name__ == "__main__":
    main()
