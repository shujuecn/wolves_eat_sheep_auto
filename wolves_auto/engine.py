"""表库引擎：加载硬解表库（v1 / v2 canonical / v2 压缩），查询任意局面
的最优结论（狼胜/羊胜/和棋 + 最短步数）与全部候选最优走法。

核心逻辑提取自 wolves_eat_sheep_hard_solve/web/server.py（Tablebase +
Session._model_choice），去掉了 HTTP/会话部分，供 CLI 与自动下棋复用。

局面输入约定（与视觉模块一致）：
  5x5 数值矩阵 grid[row][col]，0=空 1=羊(s) 2=狼(W)；row 0 在上，col 0 在左。
  棋盘坐标名 = 列字母(A-E) + 行号(1-5)，A1 为左上角。
"""
from __future__ import annotations

import copy
import mmap
import os
import random
import re
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_HARD_SOLVE_DIR = _HERE.parent / "wolves_eat_sheep_hard_solve"
for p in (str(_HARD_SOLVE_DIR), str(_HARD_SOLVE_DIR / "wolves_eat_sheep_game")):
    if p not in sys.path:
        sys.path.insert(0, p)

import hard_solve_fast as hsf  # noqa: E402
from rules import GameState, WOLF, SHEEP, IDLE_LIMIT, MAX_MOVES  # noqa: E402

EMPTY = 0
GRID_SHEEP = 1
GRID_WOLF = 2

TB_RESULT_NAME = {hsf.WOLF_WIN: "狼胜", hsf.SHEEP_WIN: "羊胜",
                  hsf.DRAW: "和棋", hsf.UNKNOWN: "未知"}

DEFAULT_TB_ROOTS = (
    _HARD_SOLVE_DIR / "data" / "ws_tb_dtc_260819",   # v1 全量
    _HARD_SOLVE_DIR / "data" / "ws_tb_dtc_v2",       # v2 canonical
    _HARD_SOLVE_DIR / "data" / "ws_tb_dtc_v2_c",     # v2 压缩
)

COLS = "ABCDE"


# ============================================================
# 坐标与局面工具
# ============================================================

def cell_name(row: int, col: int) -> str:
    """(row, col) -> 棋盘坐标名，如 (4,2) -> C5。"""
    return f"{COLS[col]}{row + 1}"


def parse_cell(name: str) -> tuple[int, int]:
    """棋盘坐标名 -> (row, col)，如 C3 -> (2, 2)。"""
    name = name.strip().upper()
    if len(name) != 2 or name[0] not in COLS or name[1] not in "12345":
        raise ValueError(f"无效坐标: {name!r}（应为 A1-E5）")
    return int(name[1]) - 1, COLS.index(name[0])


def move_name(src: tuple[int, int], dst: tuple[int, int]) -> str:
    return f"{cell_name(*src)}-{cell_name(*dst)}"


def capture_name(src: tuple[int, int], dst: tuple[int, int]) -> str:
    return f"{cell_name(*src)}x{cell_name(*dst)}"


def result_label(result: int, dist: int) -> str:
    """表库条目 -> 中文标签：狼胜·最快9步 / 和棋。"""
    lab = TB_RESULT_NAME.get(result, "未知")
    if result in (hsf.WOLF_WIN, hsf.SHEEP_WIN):
        lab += f"·最快{dist}步"
    return lab


def grid_to_fen(grid, turn: str) -> str:
    """数值矩阵 + 回合('w'/'s' 或 WOLF/SHEEP) -> FEN。"""
    rows = []
    for r in range(5):
        row, empty = "", 0
        for c in range(5):
            v = grid[r][c]
            if v == EMPTY:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += "W" if v == GRID_WOLF else "s"
        if empty:
            row += str(empty)
        rows.append(row)
    t = turn
    if t in (WOLF, SHEEP):
        t = "w" if t == WOLF else "s"
    elif t == GRID_WOLF:
        t = "w"
    elif t == GRID_SHEEP:
        t = "s"
    return "/".join(rows) + " " + t


def fen_to_grid(fen: str) -> tuple[list, str]:
    """FEN -> (数值矩阵, 'w'|'s')。

    空格可用数字（web 版约定）或 '.'：'sssss/sssss/sssss/5/1w2w1 w' 或
    'sssss/sssss/sssss/...../.www. w' 等价。
    """
    parts = fen.split()
    rows = parts[0].split("/")
    if len(rows) != 5:
        raise ValueError(f"FEN 行数错误: {fen!r}")
    grid = [[0] * 5 for _ in range(5)]
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            elif ch == ".":
                c += 1
            elif ch in "wW":
                grid[r][c] = GRID_WOLF
                c += 1
            elif ch in "sS":
                grid[r][c] = GRID_SHEEP
                c += 1
            else:
                raise ValueError(f"FEN 非法字符 {ch!r}")
        if c != 5:
            raise ValueError(f"FEN 行宽错误: {row!r}")
    turn = "w"
    if len(parts) > 1:
        t = parts[1].lower()
        if t not in ("w", "s"):
            raise ValueError(f"FEN 回合应为 w/s: {parts[1]!r}")
        turn = t
    return grid, turn


def board_str(grid) -> str:
    """数值矩阵 -> 25 字符串（. s W），便于命令行输入。"""
    m = {EMPTY: ".", GRID_SHEEP: "s", GRID_WOLF: "W"}
    return "".join(m[grid[r][c]] for r in range(5) for c in range(5))


def board_from_str(s: str):
    """25 字符串 -> 数值矩阵（按行填充，大小写均可）。"""
    s = re.sub(r"\s+", "", s).replace("/", "").lower()
    if len(s) != 25 or any(ch not in ".sw" for ch in s):
        raise ValueError(f"棋盘串应为 25 个 .sW 字符: {s!r}")
    m = {".": EMPTY, "s": GRID_SHEEP, "w": GRID_WOLF}
    return [[m[s[r * 5 + c]] for c in range(5)] for r in range(5)]


def render_grid(grid, turn=None, title: str | None = None) -> str:
    """ASCII 渲染 5x5 局面（含 A-E/1-5 坐标）。"""
    lines = []
    if title:
        lines.append(title)
    lines.append("    A   B   C   D   E")
    lines.append("  ┌───┬───┬───┬───┬───┐")
    sym = {EMPTY: " ", GRID_SHEEP: "s", GRID_WOLF: "W"}
    for r in range(5):
        cells = " │ ".join(sym[grid[r][c]] for c in range(5))
        lines.append(f"{r + 1} │ {cells} │")
        if r < 4:
            lines.append("  ├───┼───┼───┼───┼───┤")
    lines.append("  └───┴───┴───┴───┴───┘")
    if turn is not None:
        who = {"w": "狼", "s": "羊"}.get(turn, turn)
        lines.append(f"轮到: {who}")
    return "\n".join(lines)


def game_from_grid(grid, turn: str, max_moves: int | None = MAX_MOVES,
                   idle_limit: int | None = IDLE_LIMIT) -> GameState:
    """由数值矩阵构造 GameState（不校验合法性，交由查表/走子逻辑处理）。"""
    g = GameState(idle_limit=idle_limit, max_moves=max_moves)
    g.board = [
        [None if grid[r][c] == EMPTY else
         (WOLF if grid[r][c] == GRID_WOLF else SHEEP)
         for c in range(5)] for r in range(5)]
    g.turn = WOLF if turn in ("w", WOLF, GRID_WOLF) else SHEEP
    return g


def game_winner_grid(grid, turn: str) -> tuple[str, str] | None:
    """按官方规则判定该局面是否已终局，返回 (结果, 原因) 或 None。"""
    g = game_from_grid(grid, turn)
    if g.sheep_count < 4:
        return "狼胜", "羊被吃到不足 4 只"
    if not any(g.legal_moves_from((r, c))
               for r in range(5) for c in range(5) if g.board[r][c] == WOLF):
        return "羊胜", "三只狼均无法移动"
    return None


def pos_key_of(game: GameState):
    """局面唯一 key（棋盘 + 轮到谁走），用于重复计数。"""
    return (tuple(game.board[i // 5][i % 5] for i in range(25)), game.turn)


# ============================================================
# 表库文件头解析（与 web/server.py 一致）
# ============================================================

TB_FILE_RE = re.compile(r"dtc_k(\d+)\.bin\Z")


def _parse_tb_header(hdr: bytes):
    if len(hdr) < 64:
        return None
    if hdr[:4] != b"WSTB" or hdr[24] != 1:
        return None
    version = hdr[4]
    flags = hdr[7]
    total_entries = struct.unpack_from("<Q", hdr, 16)[0]
    aux_len = struct.unpack_from("<I", hdr, 28)[0]
    chunk_count = struct.unpack_from("<I", hdr, 32)[0]
    cdata_len = struct.unpack_from("<I", hdr, 36)[0]
    return version, flags, total_entries, aux_len, chunk_count, cdata_len


def tb_file_completed(path: str, k: int) -> bool:
    try:
        with open(path, "rb") as f:
            hdr = f.read(64)
        info = _parse_tb_header(hdr)
        if info is None:
            return False
        version, flags, total_entries, aux_len, cc, cdlen = info
        if (flags & 2) and version == 2 and (flags & 1):
            expect = 64 + cdlen + 12 * cc + aux_len
        elif version == 2 and (flags & 1):
            expect = 64 + total_entries + aux_len
        else:
            expect = 64 + hsf.bucket_size(k)
        return os.path.getsize(path) == expect
    except OSError:
        return False


def find_tb_dir(explicit: str | os.PathLike | None = None) -> str:
    """定位表库目录：显式参数优先，否则在默认位置中选“已截完最大 k”最大的目录。"""
    if explicit is not None:
        d = Path(explicit)
        if not d.is_dir():
            raise FileNotFoundError(f"表库目录不存在: {d}")
        return str(d)
    best_dir, best_k = None, 3
    for root in DEFAULT_TB_ROOTS:
        k = max_completed_k(str(root))
        if k > best_k:
            best_dir, best_k = str(root), k
    if best_dir is None:
        raise FileNotFoundError(
            "未找到已解出的表库目录，请先求解或用 --tb-dir 指定 "
            f"（尝试过: {', '.join(str(r) for r in DEFAULT_TB_ROOTS)}）")
    return best_dir


def max_completed_k(data_dir: str) -> int:
    best = 3
    try:
        names = os.listdir(data_dir)
    except OSError:
        return best
    for name in names:
        m = TB_FILE_RE.match(name)
        if not m:
            continue
        k = int(m.group(1))
        if k > best and tb_file_completed(os.path.join(data_dir, name), k):
            best = k
    return best


# ============================================================
# 表库（mmap 只读；支持 v1 全量 / v2 canonical / v2 deflate 分块压缩）
# ============================================================

class Tablebase:
    def __init__(self, data_dir: str):
        self.dir = str(data_dir)
        self.max_k = max_completed_k(self.dir)
        self._mm = {}
        self._canon = {}
        self._entries = {}
        self._chunks = {}
        self._cdata = {}
        self._cached = {}
        self._lru = {}

    def _open(self, k: int) -> bool:
        if k in self._mm:
            return True
        path = os.path.join(self.dir, f"dtc_k{k:02d}.bin")
        try:
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                info = _parse_tb_header(f.read(64))
                if info is None:
                    return False
                version, flags, total_entries, aux_len, cc, cdlen = info
                if (flags & 2) and version == 2 and (flags & 1):
                    expect = 64 + cdlen + 12 * cc + aux_len
                elif version == 2 and (flags & 1):
                    expect = 64 + total_entries + aux_len
                else:
                    expect = 64 + hsf.bucket_size(k)
                if size != expect:
                    return False
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError:
            return False
        compressed = (flags & 2) != 0
        if version == 2 and (flags & 1):
            if compressed:
                chunk_count, cdata_len = struct.unpack_from("<2I", mm, 32)
                idx_off = 64 + cdata_len
                chunks = []
                for i in range(chunk_count):
                    off = struct.unpack_from("<Q", mm, idx_off + 12 * i)[0]
                    sz = struct.unpack_from("<I", mm, idx_off + 12 * i + 8)[0]
                    chunks.append((off, sz))
                self._chunks[k] = chunks
                self._cdata[k] = (cdata_len, chunk_count)
                self._cached[k] = {}
                self._lru[k] = []
                auxt = mm[idx_off + 12 * chunk_count:
                          idx_off + 12 * chunk_count + aux_len]
            else:
                auxt = mm[64 + total_entries: 64 + total_entries + aux_len]
            canon = hsf.CanonIndex()
            if not canon.parse_aux(auxt, mm[5]):
                mm.close()
                return False
            self._canon[k] = canon
            self._entries[k] = total_entries
        else:
            self._canon[k] = None
            self._entries[k] = hsf.bucket_size(k)
        self._mm[k] = mm
        return True

    def _chunk_bytes(self, k: int, c: int) -> bytes:
        import zlib
        cache = self._cached.get(k)
        if c in cache:
            lru = self._lru[k]
            if c in lru:
                lru.remove(c)
            lru.append(c)
            return cache[c]
        off, sz = self._chunks[k][c]
        raw = zlib.decompress(self._mm[k][off:off + sz])
        cache[c] = raw
        self._lru[k].append(c)
        while len(cache) > 256:
            victim = self._lru[k].pop(0)
            cache.pop(victim, None)
        return raw

    def _entry_at(self, k: int, slot: int) -> int:
        if self._chunks.get(k) is None:
            return self._mm[k][64 + slot]
        c = slot // hsf.CHUNK_ENTRIES
        buf = self._chunk_bytes(k, c)
        return buf[slot - c * hsf.CHUNK_ENTRIES]

    def lookup_ranks(self, wr: int, sr: int, k: int, turn: bool):
        if k < 4:
            return True, hsf.WOLF_WIN, 0
        if not self._open(k):
            return False, hsf.UNKNOWN, 0
        canon = self._canon.get(k)
        idx = (canon.state_slot(wr, sr, turn) if canon is not None
               else hsf.state_index(wr, sr, k, turn))
        entry = self._entry_at(k, idx)
        return True, entry & 3, (entry >> 2) & 0x3F

    def lookup_game(self, game: GameState):
        """按 GameState 查表，返回 (known, result, dist)。"""
        k = game.sheep_count
        if k < 4:
            return True, hsf.WOLF_WIN, 0
        if not self._open(k):
            return False, hsf.UNKNOWN, 0
        gr = hsf.encode_wolf([i for i in range(25) if game.board[i // 5][i % 5] == WOLF])
        sr = hsf.encode_sheep([i for i in range(25) if game.board[i // 5][i % 5] == SHEEP],
                              gr, k)
        return self.lookup_ranks(gr, sr, k, game.turn == SHEEP)

    def lookup(self, grid, turn: str):
        """便捷入口：数值矩阵 + 回合直接查表。"""
        return self.lookup_game(game_from_grid(grid, turn))

    def verdict(self, grid, turn: str) -> dict:
        known, result, dist = self.lookup(grid, turn)
        winner = game_winner_grid(grid, turn)
        if winner is not None:
            return {"known": True, "result": None, "dist": 0,
                    "label": f"{winner[0]}·终局", "terminal": winner,
                    "sheep": sum(v == GRID_SHEEP for row in grid for v in row)}
        if not known:
            return {"known": False, "label": f"表库 k={sum(v == GRID_SHEEP for row in grid for v in row)} 未求解",
                    "terminal": None}
        return {"known": True, "result": result, "dist": dist,
                "label": result_label(result, dist), "terminal": None}


# ============================================================
# 引擎：候选走法评分与选步
# ============================================================

class Engine:
    """表库引擎：给定局面给出全部候选走法的表库结论，并按策略选一步。

    选步策略（与 web 版一致）：
      - 只在当前回合方最优档内选（必胜取最快、必败拖延最久、和棋保和）；
      - 同档内按 DTM 最优取前三条随机；重复局面和步数上限由状态机判和；
    """

    def __init__(self, tb_dir: str | os.PathLike | None = None):
        self.tb_dir = find_tb_dir(tb_dir)
        self.tb = Tablebase(self.tb_dir)

    @property
    def max_k(self) -> int:
        return self.tb.max_k

    # ---- 单步评估 ----
    def evaluate_move(self, grid, turn: str, src: tuple[int, int],
                      dst: tuple[int, int]) -> dict | None:
        """评估一个走法：走完后局面的表库结论。非法返回 None。"""
        g = game_from_grid(grid, turn)
        if g.board[src[0]][src[1]] != g.turn:
            return None
        mv = next((m for m in g.legal_moves_from(src) if m.destination == dst), None)
        if mv is None:
            return None
        sheep_before = g.sheep_count
        if not g.move(src, dst):
            return None
        captured = dst if g.sheep_count < sheep_before else None
        nxt_grid = [[{None: EMPTY, WOLF: GRID_WOLF, SHEEP: GRID_SHEEP}[g.board[r][c]]
                     for c in range(5)] for r in range(5)]
        nxt_turn = SHEEP if turn in ("w", WOLF, GRID_WOLF) else WOLF
        verdict = self.tb.verdict(nxt_grid, nxt_turn)
        return {
            "src": src, "dst": dst, "captured": captured,
            "mover": turn, "verdict": verdict,
            "rank": verdict_rank(verdict, turn),
            "label": verdict["label"],
            "name": (capture_name if captured else move_name)(src, dst),
            "next_key": pos_key_of(g),
        }

    # ---- 全部合法走法评分 ----
    def ranked_moves(self, grid, turn: str) -> list[dict]:
        """当前回合方全部合法走法，按对走子方有利程度排序。

        排序：胜(rank2, 快者优先) > 和(rank1) > 负(rank0, 拖得久者优先)。
        """
        g = game_from_grid(grid, turn)
        out = []
        for r in range(5):
            for c in range(5):
                if g.board[r][c] != g.turn:
                    continue
                for mv in g.legal_moves_from((r, c)):
                    ev = self.evaluate_move(grid, turn, (r, c), mv.destination)
                    if ev is not None:
                        out.append(ev)
        out.sort(key=lambda m: (m["rank"],
                                -m["verdict"]["dist"] if m["rank"] == 2
                                else (m["verdict"]["dist"] if m["rank"] == 0 else 0)),
                 reverse=True)
        return out

    # ---- 选步 ----
    def pick_move(self, grid, turn: str, seen: set | None = None) -> dict | None:
        """从最优档的前三条候选中随机选一步，返回走法 dict 或 None。"""
        current = self.tb.verdict(grid, turn)
        required_result = verdict_result_code(current)
        ranked = self.ranked_moves(grid, turn)
        pool = shortlist_optimal_moves(ranked, seen=seen,
                                       required_result=required_result)
        if not pool:
            return None
        chosen = copy.deepcopy(random.choice(pool))
        chosen["pool_size"] = len(pool)
        return chosen


def verdict_rank(verdict: dict, turn: str) -> int:
    """结论对走子方的有利度：2 胜 / 1 和 / 0 负。"""
    res = verdict.get("result")
    my_win = hsf.WOLF_WIN if turn in ("w", WOLF, GRID_WOLF) else hsf.SHEEP_WIN
    if verdict.get("terminal"):
        term_side = verdict["terminal"][0]
        return 2 if (("狼" in term_side) == (turn in ("w", WOLF, GRID_WOLF))) else 0
    if not verdict.get("known"):
        return -1
    return 2 if res == my_win else (1 if res == hsf.DRAW else 0)


def shortlist_optimal_moves(ranked: list[dict], seen: set | None = None,
                            limit: int = 3, required_result: int | None = None) -> list[dict]:
    """返回满足结果硬约束和最优排序的候选池。"""
    if not ranked or limit <= 0:
        return []
    if required_result is not None:
        same_result = [m for m in ranked
                       if verdict_result_code(m["verdict"]) == required_result]
        if not same_result:
            return []
        ranked = same_result
    best_rank = ranked[0]["rank"]
    pool = [m for m in ranked if m["rank"] == best_rank]

    if seen:
        fresh = [m for m in pool if m["next_key"] not in seen]
        if fresh:
            pool = fresh

    if best_rank == 2:
        pool.sort(key=lambda m: m["verdict"]["dist"])
    elif best_rank == 0:
        pool.sort(key=lambda m: m["verdict"]["dist"], reverse=True)
    return pool[:limit]


def verdict_result_code(verdict: dict) -> int | None:
    """统一普通表库结果和立即终局结果的编码。"""
    result = verdict.get("result")
    if result is not None:
        return result
    terminal = verdict.get("terminal")
    if terminal:
        return {"狼胜": hsf.WOLF_WIN, "羊胜": hsf.SHEEP_WIN}.get(terminal[0])
    return None


# ============================================================
# 展示辅助（CLI 与交互菜单共用）
# ============================================================

def best_report_lines(eng: "Engine", grid, turn: str, top: int = 8) -> list[str]:
    """生成最优解查询报告（ASCII 棋盘 + 结论 + 候选走法排序）。"""
    lines = [render_grid(grid, turn)]
    k = sum(v == GRID_SHEEP for row in grid for v in row)
    verdict = eng.tb.verdict(grid, turn)
    lines.append(f"表库: {eng.tb_dir} (max_k={eng.max_k})  羊数 k={k}")
    if verdict["terminal"]:
        res, reason = verdict["terminal"]
        lines.append(f"\n对局已终局: {res}（{reason}）")
        return lines
    lines.append(f"结论: {verdict['label']}")
    moves = eng.ranked_moves(grid, turn)
    if not moves:
        lines.append("无合法走法（或表库不可用）。")
        return lines
    shown = moves if top <= 0 else moves[:top]
    side_cn = "狼" if turn in ("w", WOLF, GRID_WOLF) else "羊"
    lines.append(f"\n候选走法（共 {len(moves)} 条，按对{side_cn}方有利排序）:")
    for i, m in enumerate(shown, 1):
        cap = f" 吃{cell_name(*m['captured'])}" if m["captured"] else ""
        star = " ←推荐" if i == 1 else ""
        lines.append(f"  {i:2d}. {m['name']}{cap:<10} -> {m['label']}{star}")
    return lines


def tb_info_lines(tb_dir: str | os.PathLike | None = None) -> list[str]:
    """生成表库信息报告（目录、覆盖范围、各桶条目数）。"""
    d = find_tb_dir(tb_dir)
    tb = Tablebase(d)
    lines = [f"表库目录: {d}",
             f"已解出最大羊数 k = {tb.max_k}"
             + ("（开局 k=15 已全覆盖）" if tb.max_k >= 15 else "")]
    total = 0
    for k in range(4, tb.max_k + 1):
        if tb._open(k):
            entries = tb._entries[k]
            total += entries
            canon = "canonical(v2)" if tb._canon.get(k) is not None else "full(v1)"
            comp = "+deflate" if k in tb._chunks else ""
            lines.append(f"  k={k:2d}: {entries:>13,} entries  [{canon}{comp}]")
    lines.append(f"合计条目: {total:,}")
    return lines
