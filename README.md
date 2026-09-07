# 狼羊棋助手：硬解、视觉检测与自动对局

本项目通过统一入口 `cli.py` 提供全屏终端界面（TUI）和脚本化命令：

1. **狼羊棋最优解**（[wolves_eat_sheep_hard_solve/](wolves_eat_sheep_hard_solve/README.md)）
   对 5×5 狼羊棋（3 狼 vs 15 羊，狼先行）做完整逆向分析，k=4…15 全部 **188 亿**局面
   的胜负结论与最短步数（DTM）已硬解入表库。本地默认使用 v2 压缩表库
   （`wolves_eat_sheep_hard_solve/data/ws_tb_dtc_v2_c`，3.7 GB）。
2. **屏幕视觉 + 鼠标操控**（`scripts/`，算法核心 `wolves_cv`）
   自动定位微信小程序「狼吃羊棋」窗口，识别 5×5 棋子分布、阵营与走棋方，
   并可模拟鼠标移动棋子（A1-E5 坐标系）。

`wolves_auto/` 包提供 TUI、表库引擎、屏幕观测、鼠标控制与自动下棋闭环。

## 安装

```bash
pip install opencv-python numpy Pillow pyautogui mss
```

macOS 需在 **系统设置 → 隐私与安全性** 中给终端授予 **辅助功能**（鼠标控制）
与 **屏幕录制**（抓屏）权限。

## 终端界面

```bash
python3 cli.py
# 等价命令
python3 cli.py tui
```

TUI 提供多级导航，使用方向键选择、`Enter` 确认，`Esc`/`B` 返回上一级：

- **自动对局**：选择自动/狼/羊、演示模式和最多行棋数，实时显示棋盘、状态和决策日志；残局询问时按 `W` 执狼或 `S` 执羊。
- **棋局工具**：进入最优解分析、屏幕检测或手动走子。
- **系统设置**：自动定位游戏窗口、手动设置窗口区域、检查或切换表库目录。

建议先启用“演示模式”确认棋盘识别和走法，再允许实际鼠标操作。自动对局中按
`Q`/`Esc` 停止并返回；pyautogui 的屏幕左上角紧急停止仍然有效。

## 自动对局逻辑

1. 连续抓取 3 帧并投票，得到稳定的 5x5 棋盘；帧间隔 50ms，主循环间隔 150ms。
2. 检测到完整初始局（3 狼、15 羊）时，根据棋盘朝向自动识别用户执棋方并直接求解。
3. 从残局启动且选择“自动”时，棋子数量确认是有效对局后询问用户执狼或执羊；若回合提示无法识别，默认从用户所选阵营的当前步接管。
4. 后续回合由实际事件交替推断：我方成功点击后轮到对手；棋盘从我方落子结果再次变化后轮回我方。界面颜色识别只用于残局首次校准，不再是每步行棋的硬前提。
5. 轮到用户执棋方时查询表库。候选子局面必须保持当前局面的全局结论：当前狼胜绝不选和棋或羊胜，当前羊胜绝不选和棋或狼胜，当前和棋只选和棋。通过硬约束后，从 DTM 最优的前三条中随机行棋。必胜取最短三条，必败取最长三条防守，和棋取前三条保和走法。
6. 我方落子后，只有从预测棋盘出发的一步合法对手着法才会推进状态；羊数突然增加、棋子凭空出现或动画中间帧都会被忽略并记录。棋盘未变化超时才重试点击，棋盘已经是我方落子结果则继续等待对手，避免连续走两步。
7. 羊少于 4 只、三狼全部无路可走、重复局面达到 5 次或达到步数上限时报告终局；检测到下一局完整初始局后自动重置阵营、回合和局面计数。
8. 显式窗口区域始终复用；自动定位成功后 60 秒内不再调用 `osascript`，减少自动对局停顿。

## CLI 用法

TUI 与命令行共用同一套引擎。脚本调用时使用以下子命令。

### 1. 查询任意局面的最优解 `best`

```bash
# FEN 输入（空格可用数字或 '.'）：初始局面，狼先行
python3 cli.py best --fen "sssss/sssss/sssss/...../.www. w"

# 25 字符棋盘串（按行，. s W，大小写均可），回合用 --turn 指定
python3 cli.py best --board "ssssssssss.s.sss..s.W.WW." --turn w

# 直接从屏幕识别当前局面再求解（自动找游戏窗口）
python3 cli.py best --screen
python3 cli.py best --screen --region "1004,38,508,944" --turn s --top 15
```

输出：ASCII 棋盘、表库结论（狼胜/羊胜/和棋·最快 N 步）、按有利度排序的
全部候选走法（含跳吃标记）。

### 2. 观测屏幕 `detect`

```bash
python3 cli.py detect                 # 单次：输出局面/阵营/走棋方
python3 cli.py detect --suggest       # 附带表库结论与双方最优建议
python3 cli.py detect --watch 60      # 持续观测 60 秒
python3 cli.py detect --image 图片.png --suggest   # 调试：识别图片而非屏幕
```

### 3. 自动下棋 `play`

识别 → 表库最优解 → 鼠标执行的完整闭环：

```bash
# 完整初始局自动判定阵营；残局会在终端询问执棋方
python3 cli.py play --side auto

# 指定阵营；演示模式只打印建议走法、不点击鼠标
python3 cli.py play --side 狼 --demo
python3 cli.py play --side 羊

# 常用参数
python3 cli.py play --side auto --region "1004,38,508,944" \
    --post-move-delay 1.5 --await-timeout 90 --max-half-moves 100
```

行为细节：

- **走棋方判断**：完整局固定狼先；残局首次使用提示或用户执棋方，后续按棋盘变化交替推断。
- **选步策略**：先锁定与当前局面相同的胜负结果，再从 DTM 最优前三条随机选择；重复局面达到 5 次或步数达到上限时判和。
- **终局与新局**：羊 <4（狼胜）、三狼全堵死（羊胜）、重复局面 5 次或达到步数上限时报告结果；
  检测到初始局面自动开新局并重置局面计数。
- **紧急停止**：`Ctrl+C`，或把鼠标甩到屏幕左上角（pyautogui FAILSAFE）。

### 4. 其他命令

```bash
python3 cli.py move C5 C4        # 手动鼠标走子（双击源棋子 + 单击目的地）
python3 cli.py move A1 B2 --demo # 演示模式
python3 cli.py window            # 打印找到的游戏窗口 (x,y,w,h)
python3 cli.py tb                # 表库目录、覆盖范围与各桶条目数
python3 cli.py tui               # 显式启动全屏终端界面
```

## 坐标系

```
    A   B   C   D   E
1 │A1 │B1 │C1 │D1 │E1 │   左上角 A1，右下角 E5
2 │A2 │B2 │C2 │D2 │E2 │   row 1-5 自上而下
3 │A3 │B3 │C3 │D3 │E3 │   col A-E 自左而右
4 │A4 │B4 │C4 │D4 │E4 │
5 │A5 │B5 │C5 │D5 │E5 │
```

## 整合架构

```
cli.py                      # 统一入口（默认 TUI；同时提供脚本化子命令）
wolves_auto/
├── tui.py                  # curses 全屏 TUI、多级菜单、后台自动对局与实时棋盘
├── engine.py               # 表库查询、候选评分与前三最优随机选步
├── screen.py               # 屏幕观测：窗口定位(osascript)、标注图网格解析、
│                           #   逐帧棋子识别(投票稳定)、回合/阵营检测（源自 live_detect.py）
├── mousectl.py             # 鼠标操控：A1-E5 → 像素坐标、双击选中+单击落子（源自 move_piece.py）
└── player.py               # 自动对局状态机：阵营/回合推断、决策、执行、等待与终局
scripts/wolves_cv/          # 视觉算法核心（棋盘定位/棋子判据/事件分析，被 screen.py 复用）
scripts/live_detect.py      # 旧版独立实时检测（保留可用）
scripts/move_piece.py       # 旧版独立鼠标走子（保留可用）
wolves_eat_sheep_hard_solve/# 求解器、表库、规则库 rules.py、Web 版（见其 README）
```

依赖关系：`engine.py` 只用标准库 + 项目内 `rules.py`/`hard_solve_fast.py`；
`screen.py`/`mousectl.py`/`player.py` 额外需要 cv2/numpy/mss/pyautogui。
因此 `best`（非 screen）、`tb` 命令在无视觉依赖的环境也可运行。

## 表库说明

- 位置：`wolves_eat_sheep_hard_solve/data/`，CLI 默认自动选择“已截完最大 k”
  的目录（本地为 `ws_tb_dtc_v2_c`，k=4…15 全覆盖，开局即有精确解）。
- `--tb-dir` 可指向任意表库目录（v1 全量 / v2 canonical / v2 压缩均自动识别）。
- 求解、校验、统计等工具见 [wolves_eat_sheep_hard_solve/README.md](wolves_eat_sheep_hard_solve/README.md)。

## 验证记录

- `python3 -m unittest discover -s tests -p 'test_*.py' -v`：验证前三候选、胜负 DTM 排序、重复计数、残局接管和未知回合交替。
- 初始局面 `best`：压缩表库返回和棋，唯一保和走法为 `C5xC3`。
- 截图识别（`materials/QQ20260824-132618.png`）：识别出
  `sssss/sssss/.s.ss/s..s./W.WW.`（W:3 S:15，与画面 HUD 一致），
  表库结论羊胜·最快20步。
- TUI 冒烟测试：验证主菜单启动、全屏绘制和 `Q` 正常退出。
