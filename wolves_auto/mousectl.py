"""鼠标操控：把棋盘坐标 (row, col) / A1-E5 映射为窗口内像素并模拟点击走子。

走子方式与 scripts/move_piece.py 一致：先双击源棋子（首次聚焦窗口、
第二次选中棋子），再单击目的地。
"""
from __future__ import annotations

import time

import pyautogui

from .engine import parse_cell, cell_name

# 从标注图解析的网格归一化坐标（与 move_piece.py 一致）
NX = [0.146, 0.323, 0.500, 0.676, 0.852]   # A-E 列
NY = [0.337, 0.429, 0.521, 0.613, 0.705]   # 1-5 行

pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True   # 鼠标甩到屏幕左上角可紧急中止


class MouseController:
    def __init__(self, demo: bool = False, logger=print):
        self.demo = demo
        self.log = logger

    def cell_pixel(self, row: int, col: int,
                   win_x: int, win_y: int, win_w: int, win_h: int):
        return win_x + int(NX[col] * win_w), win_y + int(NY[row] * win_h)

    def click_cell(self, row: int, col: int, region: tuple[int, int, int, int],
                   clicks: int = 1):
        x, y = self.cell_pixel(row, col, *region)
        if self.demo:
            self.log(f"演示 | 点击 {cell_name(row, col)} @ ({x}, {y}) x{clicks}")
            return
        pyautogui.click(x, y, clicks=clicks, interval=0.15)

    def move_piece(self, src: tuple[int, int], dst: tuple[int, int],
                   region: tuple[int, int, int, int], settle: float = 0.8) -> bool:
        """从 src 走到 dst（棋盘坐标）。demo 模式只打印。"""
        sr, sc = src
        dr_, dc = dst
        name = f"{cell_name(sr, sc)} -> {cell_name(dr_, dc)}"
        sx, sy = self.cell_pixel(sr, sc, *region)
        dx, dy = self.cell_pixel(dr_, dc, *region)
        self.log(f"执行 | {name}  ({sx}, {sy}) -> ({dx}, {dy})")
        if self.demo:
            self.log("演示 | 未执行鼠标操作")
            return False
        pyautogui.click(sx, sy)
        time.sleep(0.15)
        pyautogui.click(sx, sy)      # 双击选中棋子
        time.sleep(0.3)
        pyautogui.click(dx, dy)      # 单击目的地
        time.sleep(settle)
        return True

    def move_named(self, src_name: str, dst_name: str,
                   region: tuple[int, int, int, int]) -> bool:
        return self.move_piece(parse_cell(src_name), parse_cell(dst_name), region)


def parse_region(text: str) -> tuple[int, int, int, int]:
    parts = tuple(int(v) for v in text.split(","))
    if len(parts) != 4:
        raise ValueError(f"--region 应为 x,y,w,h: {text!r}")
    return parts
