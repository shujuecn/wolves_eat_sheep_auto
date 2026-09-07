"""基于 curses 的统一终端界面。"""
from __future__ import annotations

import curses
import queue
import threading
import time
from dataclasses import dataclass

from .engine import Engine, best_report_lines, tb_info_lines
from .mousectl import MouseController, parse_region
from .player import AutoPlayer
from .screen import ScreenObserver, find_game_window


@dataclass
class SideRequest:
    event: threading.Event
    value: str | None = None


class TuiApp:
    def __init__(self, screen):
        self.screen = screen
        self.settings = {
            "side": "auto",
            "demo": False,
            "max_moves": 150,
            "region": None,
            "tb_dir": None,
            "post_move_delay": 0.8,
            "await_timeout": 60.0,
        }
        self._engine = None
        self._init_screen()

    @property
    def engine(self):
        if self._engine is None:
            self.engine_status = "正在加载表库..."
            self._engine = Engine(self.settings["tb_dir"])
        return self._engine

    def _init_screen(self):
        curses.curs_set(0)
        self.screen.keypad(True)
        self.screen.timeout(-1)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
            curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)

    def run(self):
        while True:
            choice = self._menu(
                "狼羊棋助手",
                ["自动对局", "棋局工具", "系统设置", "退出"],
                "↑↓ 选择  Enter 确认  Q 退出",
                allow_back=False,
            )
            if choice == 0:
                self._auto_setup()
            elif choice == 1:
                self._tools_menu()
            elif choice == 2:
                self._system_menu()
            else:
                return

    def _menu(self, title, items, hint="Esc/B 返回", allow_back=True):
        selected = 0
        while True:
            self.screen.erase()
            height, width = self.screen.getmaxyx()
            self._add(1, 3, title, curses.A_BOLD | self._color(3))
            self._add(2, 3, "─" * max(10, min(width - 6, 54)), self._color(3))
            top = max(4, (height - len(items) * 2) // 2)
            for index, item in enumerate(items):
                marker = "  > " if index == selected else "    "
                style = curses.A_REVERSE | curses.A_BOLD if index == selected else 0
                self._add(top + index * 2, 5, marker + item, style)
            self._footer(hint)
            self.screen.refresh()
            key = self.screen.getch()
            if key in (curses.KEY_UP, ord("k"), ord("K")):
                selected = (selected - 1) % len(items)
            elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
                selected = (selected + 1) % len(items)
            elif key in (10, 13, curses.KEY_ENTER):
                return selected
            elif allow_back and key in (27, ord("b"), ord("B"), ord("q"), ord("Q")):
                return None
            elif not allow_back and key in (ord("q"), ord("Q")):
                return len(items) - 1

    def _auto_setup(self):
        while True:
            maximum = self.settings["max_moves"] or "不限"
            choice = self._menu(
                "自动对局 / 设置",
                [f"执棋方        {self.settings['side']}",
                 f"演示模式      {'开' if self.settings['demo'] else '关'}",
                 f"最多行棋      {maximum}",
                 "开始对局"],
            )
            if choice is None:
                return
            if choice == 0:
                sides = ["auto", "狼", "羊"]
                current = sides.index(self.settings["side"])
                self.settings["side"] = sides[(current + 1) % len(sides)]
            elif choice == 1:
                self.settings["demo"] = not self.settings["demo"]
            elif choice == 2:
                raw = self._input("最多执行步数，留空表示不限", "")
                if raw is not None:
                    if not raw:
                        self.settings["max_moves"] = None
                    elif raw.isdigit() and int(raw) > 0:
                        self.settings["max_moves"] = int(raw)
                    else:
                        self._message("参数错误", "请输入正整数。")
            else:
                self._auto_session()

    def _auto_session(self):
        events = queue.Queue()
        stop = threading.Event()
        grid = [[0] * 5 for _ in range(5)]
        info = "正在启动"
        logs = []
        side_request = None

        def prompt_side(_obs):
            request = SideRequest(threading.Event())
            events.put(("side", request))
            while not stop.is_set() and not request.event.wait(0.1):
                pass
            return request.value

        def worker():
            try:
                player = AutoPlayer(
                    self.engine,
                    side=self.settings["side"],
                    demo=self.settings["demo"],
                    region=self.settings["region"],
                    post_move_delay=self.settings["post_move_delay"],
                    await_timeout=self.settings["await_timeout"],
                    max_half_moves=self.settings["max_moves"],
                    side_prompt=prompt_side,
                    logger=lambda line: events.put(("log", line)),
                    on_observation=lambda obs: events.put(("observation", obs)),
                )
                player.run(should_stop=stop.is_set)
            except Exception as exc:
                events.put(("log", f"错误 | {exc}"))
            finally:
                events.put(("stopped",))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        stopped = False
        self.screen.timeout(100)
        try:
            while True:
                try:
                    while True:
                        event = events.get_nowait()
                        if event[0] == "log":
                            logs.append(f"{time.strftime('%H:%M:%S')}  {event[1]}")
                            logs = logs[-200:]
                        elif event[0] == "observation":
                            obs = event[1]
                            grid = obs.grid
                            info = (f"狼 {obs.wolves}  羊 {obs.sheep}  "
                                    f"提示回合 {obs.turn or '未知'}")
                        elif event[0] == "side":
                            side_request = event[1]
                        elif event[0] == "stopped":
                            stopped = True
                except queue.Empty:
                    pass

                self._render_auto(grid, info, logs, side_request, stopped)
                key = self.screen.getch()
                if side_request is not None and key in (ord("w"), ord("W"), ord("s"), ord("S")):
                    side_request.value = "狼" if key in (ord("w"), ord("W")) else "羊"
                    side_request.event.set()
                    side_request = None
                elif key in (27, ord("q"), ord("Q")):
                    stop.set()
                    if side_request is not None:
                        side_request.event.set()
                        side_request = None
                    if stopped:
                        return
                elif stopped and key != -1:
                    return
        finally:
            stop.set()
            self.screen.timeout(-1)

    def _render_auto(self, grid, info, logs, side_request, stopped):
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        self._add(0, 2, "自动对局", curses.A_BOLD | self._color(3))
        self._add(1, 2, info, self._color(2))
        self._draw_board(3, 3, grid)

        log_x = 31 if width >= 78 else 2
        log_y = 3 if width >= 78 else 17
        log_height = max(3, height - log_y - 3)
        self._add(log_y - 1, log_x, "运行日志", curses.A_BOLD)
        for row, line in enumerate(logs[-log_height:]):
            self._add(log_y + row, log_x, line)

        if side_request is not None:
            prompt = "检测到残局：按 W 执狼，按 S 执羊"
            self._add(max(0, height - 3), 2, prompt,
                      curses.A_BOLD | self._color(4))
        elif stopped:
            self._add(max(0, height - 3), 2, "对局已停止，按任意键返回", self._color(4))
        self._footer("Q/Esc 停止并返回")
        self.screen.refresh()

    def _draw_board(self, top, left, grid):
        self._add(top, left + 3, "A   B   C   D   E", curses.A_BOLD)
        for row in range(5):
            y = top + 2 + row * 2
            self._add(y, left, str(row + 1), curses.A_BOLD)
            for col in range(5):
                value = int(grid[row][col])
                symbol = "W" if value == 2 else "S" if value == 1 else "."
                color = self._color(1 if value == 2 else 5 if value == 1 else 0)
                self._add(y, left + 3 + col * 4, symbol, curses.A_BOLD | color)
            if row < 4:
                self._add(y + 1, left + 3, "|   |   |   |   |", self._color(4))

    def _tools_menu(self):
        while True:
            choice = self._menu("棋局工具", ["最优解分析", "屏幕检测", "手动走子"])
            if choice is None:
                return
            try:
                if choice == 0:
                    self._analyze()
                elif choice == 1:
                    self._detect()
                else:
                    self._manual_move()
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                self._message("操作失败", str(exc))

    def _analyze(self):
        default = "sssss/sssss/sssss/...../.www. w"
        fen = self._input("输入 FEN", default)
        if fen is None:
            return
        from .engine import fen_to_grid
        grid, turn = fen_to_grid(fen)
        self._show_text("最优解分析", best_report_lines(self.engine, grid, turn, 10))

    def _detect(self):
        self._progress("正在检测屏幕...")
        obs = ScreenObserver(region=self.settings["region"], logger=lambda _line: None).observe(
            interval=0.05)
        grid = [[int(value) for value in row] for row in obs.grid]
        lines = [f"狼 {obs.wolves}  羊 {obs.sheep}",
                 f"提示回合: {obs.turn or '未知'}",
                 f"用户执棋: {obs.faction or '残局待确认'}", ""]
        if obs.turn:
            turn = "w" if obs.turn == "狼" else "s"
            lines.extend(best_report_lines(self.engine, grid, turn, 5))
        else:
            for turn, label in (("w", "狼方"), ("s", "羊方")):
                move = self.engine.pick_move(grid, turn)
                lines.append(f"{label}建议: {move['name']} / {move['label']}" if move
                             else f"{label}建议: 无合法走法")
        self._show_text("屏幕检测", lines)

    def _manual_move(self):
        src = self._input("起点坐标 A1-E5", "C5")
        if src is None:
            return
        dst = self._input("终点坐标 A1-E5", "C4")
        if dst is None:
            return
        actual = self._confirm("执行实际鼠标操作？默认仅演示")
        region = self.settings["region"] or find_game_window(lambda _line: None)
        if region is None:
            raise RuntimeError("未找到游戏窗口，请先在系统设置中定位。")
        lines = []
        MouseController(demo=not actual, logger=lines.append).move_named(src, dst, region)
        self._show_text("手动走子", lines)

    def _system_menu(self):
        while True:
            region = self.settings["region"] or "自动"
            tb_dir = self.settings["tb_dir"] or "自动"
            choice = self._menu(
                "系统设置",
                [f"定位游戏窗口  {region}",
                 "手动设置窗口区域",
                 f"表库目录        {tb_dir}",
                 "检查表库"],
            )
            if choice is None:
                return
            try:
                if choice == 0:
                    self._progress("正在定位游戏窗口...")
                    found = find_game_window(lambda _line: None)
                    if found is None:
                        raise RuntimeError("未找到游戏窗口。")
                    self.settings["region"] = found
                    self._message("定位完成", str(found))
                elif choice == 1:
                    raw = self._input("窗口区域 x,y,w,h，留空恢复自动", "")
                    if raw is not None:
                        self.settings["region"] = parse_region(raw) if raw else None
                elif choice == 2:
                    raw = self._input("表库目录，留空使用自动目录", "")
                    if raw is not None:
                        self.settings["tb_dir"] = raw or None
                        self._engine = None
                else:
                    self._progress("正在检查表库...")
                    self._show_text("表库信息", tb_info_lines(self.settings["tb_dir"]))
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                self._message("操作失败", str(exc))

    def _input(self, prompt, default=""):
        self.screen.erase()
        self._add(2, 3, prompt, curses.A_BOLD)
        if default:
            self._add(4, 3, f"默认: {default}", self._color(3))
        self._add(6, 3, "> ")
        self._footer("Enter 确认；直接回车使用默认值")
        self.screen.refresh()
        curses.echo()
        curses.curs_set(1)
        try:
            raw = self.screen.getstr(6, 5, 500).decode("utf-8").strip()
        except (KeyboardInterrupt, UnicodeDecodeError):
            return None
        finally:
            curses.noecho()
            curses.curs_set(0)
        return raw or default

    def _confirm(self, prompt):
        self.screen.erase()
        self._add(2, 3, prompt, curses.A_BOLD | self._color(4))
        self._add(4, 3, "Y 确认，其他键取消")
        self.screen.refresh()
        return self.screen.getch() in (ord("y"), ord("Y"))

    def _show_text(self, title, lines):
        offset = 0
        lines = [str(line) for line in lines]
        while True:
            self.screen.erase()
            height, _ = self.screen.getmaxyx()
            self._add(1, 3, title, curses.A_BOLD | self._color(3))
            visible = max(1, height - 5)
            for row, line in enumerate(lines[offset:offset + visible]):
                self._add(3 + row, 3, line)
            self._footer("↑↓ 滚动  Esc/B 返回")
            self.screen.refresh()
            key = self.screen.getch()
            if key in (27, ord("b"), ord("B"), ord("q"), ord("Q")):
                return
            if key in (curses.KEY_DOWN, ord("j")) and offset < max(0, len(lines) - visible):
                offset += 1
            elif key in (curses.KEY_UP, ord("k")) and offset > 0:
                offset -= 1

    def _message(self, title, message):
        self._show_text(title, str(message).splitlines() + ["", "按 Esc/B 返回"])

    def _progress(self, message):
        self.screen.erase()
        self._add(2, 3, message, curses.A_BOLD | self._color(3))
        self.screen.refresh()

    def _footer(self, text):
        height, _ = self.screen.getmaxyx()
        self._add(max(0, height - 1), 2, text, curses.A_DIM)

    def _add(self, y, x, text, style=0):
        height, width = self.screen.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        try:
            self.screen.addnstr(y, x, str(text), max(0, width - x - 1), style)
        except curses.error:
            pass

    @staticmethod
    def _color(pair):
        return curses.color_pair(pair) if pair and curses.has_colors() else 0


def main():
    curses.wrapper(lambda screen: TuiApp(screen).run())


if __name__ == "__main__":
    main()
