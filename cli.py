#!/usr/bin/env python3
"""wolves CLI — 狼羊棋硬解求解 + 屏幕视觉 + 自动下棋 整合入口。

不带参数运行（或子命令 tui）= 启动全屏终端界面。

子命令（脚本/非交互场景）：
  best     查询指定局面（FEN / 棋盘串 / 屏幕识别）的最优结论与候选走法
  detect   观测当前屏幕：输出识别到的局面、阵营、走棋方与表库结论
  play     全自动下棋：识别 -> 表库最优解 -> 鼠标执行（demo 只建议）
  move     手动模拟鼠标移动一枚棋子（A1-E5 坐标）
  window   打印找到的游戏窗口矩形
  tb       显示表库目录与覆盖范围

示例：
  # 终端界面
  python3 cli.py
  # 最优解：初始局面（狼先行）
  python3 cli.py best --fen "sssss/sssss/sssss/...../.www. w"
  # 从屏幕识别当前局面并给出最优解
  python3 cli.py best --screen
  # 单帧检测屏幕（含结论）
  python3 cli.py detect
  # 自动下棋（自动判阵营；--demo 只打印不点击）
  python3 cli.py play --side auto
  python3 cli.py play --side 羊 --demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wolves_auto import engine as eng_mod  # noqa: E402
from wolves_auto.engine import (Engine, board_from_str, cell_name,  # noqa: E402
                                fen_to_grid, render_grid)


def add_common(ap: argparse.ArgumentParser):
    ap.add_argument("--tb-dir", default=None,
                    help="表库目录（默认自动在 hard_solve/data 下选择已截完最大 k 的目录）")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cli.py",
                                 description="狼羊棋：硬解最优解 + 屏幕识别 + 自动下棋")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tui", help="启动全屏终端界面")
    p.set_defaults(func=cmd_tui)

    p = sub.add_parser("best", help="查询局面的最优结论与候选走法")
    add_common(p)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fen", help="局面 FEN，如 'sssss/sssss/sssss/...../.www. w'")
    src.add_argument("--board", help="25 字符棋盘串（按行，. s W）+ 可选回合，如 ssss s... W 或串尾加 w/s")
    src.add_argument("--screen", action="store_true", help="从屏幕识别当前局面")
    p.add_argument("--turn", choices=["w", "s"], help="轮到谁走（board 输入必填；screen 可覆盖）")
    p.add_argument("--region", help="游戏窗口 x,y,w,h（screen 用；默认自动找窗口）")
    p.add_argument("--top", type=int, default=8, help="展示前 N 条候选走法（0=全部）")
    p.set_defaults(func=cmd_best)

    p = sub.add_parser("detect", help="观测屏幕并输出局面/阵营/回合/结论")
    add_common(p)
    p.add_argument("--image", help="改用图片文件代替屏幕抓取（调试用）")
    p.add_argument("--region", help="游戏窗口 x,y,w,h（默认自动找窗口）")
    p.add_argument("--watch", type=float, default=0, metavar="SEC",
                   help="持续观测 SEC 秒（默认单次）")
    p.add_argument("--suggest", action="store_true", help="附带表库最优解建议")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("play", help="自动下棋（识别 -> 最优解 -> 鼠标执行）")
    add_common(p)
    p.add_argument("--side", default="auto", choices=["auto", "狼", "羊"],
                   help="我方阵营：auto=由初始局面自动判定（默认）")
    p.add_argument("--demo", action="store_true", help="演示模式：只打印建议走法，不点击鼠标")
    p.add_argument("--region", help="游戏窗口 x,y,w,h（默认自动找窗口）")
    p.add_argument("--duration", type=float, default=None, metavar="SEC",
                   help="运行秒数（默认一直运行，Ctrl+C 停止）")
    p.add_argument("--post-move-delay", type=float, default=1.2,
                   help="我方走完后等待动画的秒数（默认 1.2）")
    p.add_argument("--await-timeout", type=float, default=60.0,
                   help="等待对手走子的超时秒数（默认 60）")
    p.add_argument("--max-half-moves", type=int, default=150,
                   help="达到多少步（双方合计）后按和棋处理（默认 150）")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("move", help="手动模拟鼠标移动棋子：SRC DST（如 C5 C4）")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--region", help="游戏窗口 x,y,w,h（默认自动找窗口）")
    p.add_argument("--demo", action="store_true", help="只显示坐标不移动")
    p.set_defaults(func=cmd_move)

    p = sub.add_parser("window", help="打印找到的游戏窗口矩形 (x,y,w,h)")
    p.set_defaults(func=cmd_window)

    p = sub.add_parser("tb", help="显示表库目录与覆盖范围")
    add_common(p)
    p.set_defaults(func=cmd_tb)
    return ap


# ============================================================
# 子命令实现
# ============================================================

def cmd_tui(_args=None):
    from wolves_auto.tui import main as tui_main
    tui_main()


def _resolve_position(args) -> tuple[list, str]:
    """解析 best 命令的输入为 (grid, 'w'|'s')。"""
    if args.fen:
        return fen_to_grid(args.fen)
    if args.board:
        raw = args.board.replace(",", "").replace("/", "")
        turn = args.turn
        if raw and raw[-1] in "wsWS" and len(raw) == 26:
            turn = turn or raw[-1].lower()
            raw = raw[:-1]
        if turn is None:
            raise SystemExit("--board 输入需要 --turn w|s 指定回合（或把 w/s 附在串尾）")
        return board_from_str(raw), turn
    # --screen
    from wolves_auto.screen import ScreenObserver
    from wolves_auto.mousectl import parse_region
    obs = ScreenObserver(region=parse_region(args.region) if args.region else None)
    o = obs.observe()
    print(o.board_ascii())
    print(f"识别: W:{o.wolves} S:{o.sheep} 轮到:{o.turn or '未知'} 阵营:{o.faction or '未判定'}")
    turn = args.turn or ("w" if o.turn == "狼" else "s" if o.turn == "羊" else None)
    if turn is None:
        raise SystemExit("未能识别走棋方，请用 --turn w|s 指定")
    return [[int(v) for v in row] for row in o.grid], turn


def cmd_best(args):
    grid, turn = _resolve_position(args)
    eng = Engine(args.tb_dir)
    print(render_grid(grid, turn))
    k = sum(v == 1 for row in grid for v in row)
    verdict = eng.tb.verdict(grid, turn)
    print(f"表库: {eng.tb_dir} (max_k={eng.max_k})  羊数 k={k}")
    if verdict["terminal"]:
        res, reason = verdict["terminal"]
        print(f"\n对局已终局: {res}（{reason}）")
        return
    print(f"结论: {verdict['label']}")
    moves = eng.ranked_moves(grid, turn)
    if not moves:
        print("无合法走法（或表库不可用）。")
        return
    top = moves if args.top <= 0 else moves[:args.top]
    print(f"\n候选走法（共 {len(moves)} 条，按对{('狼' if turn == 'w' else '羊')}方有利排序）:")
    for i, m in enumerate(top, 1):
        cap = f" 吃{cell_name(*m['captured'])}" if m["captured"] else ""
        star = " ←推荐" if i == 1 else ""
        print(f"  {i:2d}. {m['name']}{cap:<10} -> {m['label']}{star}")


def cmd_detect(args):
    from wolves_auto.screen import ScreenObserver, Observation
    from wolves_auto.mousectl import parse_region

    observer = None
    if not args.image:
        observer = ScreenObserver(region=parse_region(args.region)
                                  if args.region else None)

    def once() -> Observation:
        if args.image:
            import cv2
            frame = cv2.imread(args.image)
            if frame is None:
                raise SystemExit(f"无法读取图片: {args.image}")
            vx, hy = observer.grid_for(frame.shape[1], frame.shape[0]) \
                if observer else (None, None)
            if vx is None:
                obs_obj = ScreenObserver(region=(0, 0, frame.shape[1], frame.shape[0]))
                vx, hy = obs_obj.grid_for(frame.shape[1], frame.shape[0])
                grid = obs_obj.classify_frame(frame, vx, hy)
                faction = obs_obj.detect_faction(grid)
            else:
                grid = observer.classify_frame(frame, vx, hy)
                faction = observer.detect_faction(grid)
            return Observation(grid=grid, turn=None, faction=faction,
                               wolves=int((grid == 2).sum()),
                               sheep=int((grid == 1).sum()),
                               region=None, frames=1)
        return observer.observe()

    eng = Engine(args.tb_dir) if args.suggest else None

    import time
    t0 = time.perf_counter()
    while True:
        o = once()
        print(render_grid([[int(v) for v in row] for row in o.grid],
                          title=f"[{time.strftime('%H:%M:%S')}] "
                                f"W:{o.wolves} S:{o.sheep} "
                                f"轮到:{o.turn or '?'} 我方:{o.faction or '?'}"))
        if eng is not None and not o.wolves and not o.sheep:
            print("(空棋盘)")
        elif eng is not None:
            g = [[int(v) for v in row] for row in o.grid]
            vt = eng.tb.verdict(g, "w")
            if vt["terminal"]:
                print(f"终局: {vt['terminal'][0]}（{vt['terminal'][1]}）")
            else:
                print(f"结论(按狼方视角查询): {vt['label']}")
                mv = eng.pick_move(g, "w")
                if mv:
                    print(f"狼方建议: {mv['name']}")
                mv = eng.pick_move(g, "s")
                if mv:
                    print(f"羊方建议: {mv['name']}")
        if not args.watch or time.perf_counter() - t0 >= args.watch:
            break
        time.sleep(max(0.0, args.watch / 5))


def cmd_play(args):
    from wolves_auto.player import AutoPlayer
    from wolves_auto.mousectl import parse_region

    def prompt_side(_obs):
        if not sys.stdin.isatty():
            raise RuntimeError("检测到残局，请用 --side 狼 或 --side 羊 指定执棋方")
        while True:
            value = input("检测到残局，请选择执棋方 [狼/羊]: ").strip()
            if value in ("狼", "羊"):
                return value

    eng = Engine(args.tb_dir)
    player = AutoPlayer(eng, side=args.side, demo=args.demo,
                        region=parse_region(args.region) if args.region else None,
                        post_move_delay=args.post_move_delay,
                        await_timeout=args.await_timeout,
                        max_half_moves=args.max_half_moves,
                        side_prompt=prompt_side)
    player.run(duration=args.duration)


def cmd_move(args):
    from wolves_auto.mousectl import MouseController, parse_region
    from wolves_auto.screen import find_game_window
    region = parse_region(args.region) if args.region else find_game_window()
    if region is None:
        raise SystemExit("未找到游戏窗口（可用 --region x,y,w,h 指定）")
    MouseController(demo=args.demo).move_named(args.src, args.dst, region)


def cmd_window(_):
    from wolves_auto.screen import find_game_window
    rect = find_game_window()
    print(rect if rect else "未找到游戏窗口")
    sys.exit(0 if rect else 1)


def cmd_tb(args):
    d = eng_mod.find_tb_dir(args.tb_dir)
    tb = eng_mod.Tablebase(d)
    print(f"表库目录: {d}")
    print(f"已解出最大羊数 k = {tb.max_k}"
          + ("（开局 k=15 已全覆盖）" if tb.max_k >= 15 else ""))
    total = 0
    for k in range(4, tb.max_k + 1):
        if tb._open(k):
            entries = tb._entries[k]
            total += entries
            canon = "canonical(v2)" if tb._canon.get(k) is not None else "full(v1)"
            comp = "+deflate" if k in tb._chunks else ""
            print(f"  k={k:2d}: {entries:>13,} entries  [{canon}{comp}]")
    print(f"合计条目: {total:,}")


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        cmd_tui()
        return
    args = build_parser().parse_args(raw_args)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n中断")
        sys.exit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
