#!/usr/bin/env bash
# tmux_sweep.sh —— 用 tmux 把训练 / sweep 挂到后台，关掉 ssh / 终端也照跑。（uv 版）
#
#   ./tmux_sweep.sh start train                      # 随手跑，落在 runs/_scratch/<自动名>/
#   ./tmux_sweep.sh start train:abl_norm/postnorm    # 具名 run，落在 runs/abl_norm/postnorm/
#   ./tmux_sweep.sh start train:abl_norm/postnorm_s0 --set seed=0
#   CONFIG=exp/big.json ./tmux_sweep.sh start train:big     # 用某个实验配置
#   ./tmux_sweep.sh start train:x --set max_iters=2000      # 多余的参数原样透传
#   ./tmux_sweep.sh start lr            # 起 lr sweep（会话 cs336-lr）
#   ./tmux_sweep.sh start bs            # 起 batch_size sweep
#   ./tmux_sweep.sh ls                  # 看所有会话 + 运行/已完成
#   ./tmux_sweep.sh attach train:abl_norm/postnorm   # 进去看（Ctrl-b 然后 d 脱离）
#   ./tmux_sweep.sh log train:abl_norm/postnorm      # tail -f 日志（Ctrl-C 只退 tail）
#   ./tmux_sweep.sh stop train:abl_norm/postnorm     # 停掉这个会话（连带杀掉 python）
#   ./tmux_sweep.sh clean               # 清掉已结束的会话
#   ./tmux_sweep.sh init-config exp/x.json   # 照 dataclass 默认值生成一份，供你裁剪
#
# 环境变量：
#   CONFIG=exp/small.json ./tmux_sweep.sh start train    # 用某个实验配置（默认不用）
#   GPU=1 ./tmux_sweep.sh start train:big                # 指定 CUDA_VISIBLE_DEVICES
#   UV=/path/to/uv ./tmux_sweep.sh start train           # uv 装在非标准位置
#
# 关于 job:tag —— 冒号后面那截就是 Config.tag，一处输入决定三件事：
#   tmux 会话名 cs336-train-abl_norm-postnorm （/ 压成 -，因为 tmux 的 target 语法
#                                              里 : 和 . 是分隔符，路径分隔也不合适）
#   本脚本的日志 logs/_tmux/train-abl_norm-postnorm.log
#   训练的全部产物 runs/abl_norm/postnorm/{metrics.jsonl,config.json,ckpt.pt}
# tag 允许带 / 用来分层：一层平铺撑不住 arm x seed，三个 arm 三个 seed 就是九个同级
# 目录，下周再来一族就只能靠字符串前缀去 glob 了。
#
# 不给 tag 会落到 runs/_scratch/<按超参自动拼的名>/，那里不做「已存在就拒绝」的检查
# ——smoke test 和调试跑不该被逼着起名字。反过来，具名 run 重名会当场报错而不是覆盖：
# 人给的名字不会自动撞，但会被人自己撞（崩了重跑、忘了上周用过），而覆盖是静默的。
#
# 只有 train 接受透传参数和 CONFIG。sweep 系脚本（lr / bs）没有命令行接口：
# 网格和基线配置都写死在各自的 .py 里，改实验就去改那个文件。这不是偷懒——
# 本脚本不解析下游的参数，传了它不认识的东西只会被 python 无声吞掉，
# 你以为换了配置其实没换，比直接报错难查得多，所以这里选择当场拒绝。
#
# 配置文件怎么组织（为什么 start train 默认不读任何 JSON）：
#   默认值的唯一事实来源是 Config dataclass —— 它在 git 里、有类型、和代码同步演进。
#   exp/*.json 不是「默认配置」，而是一个个具名实验，内容应该只有相对默认值的差异。
#   如果让 start train 默认去读 exp/base.json，就有了两份默认值：改了 dataclass 而
#   忘了改 base.json，跑出来的其实是旧默认值，而且 python train.py 裸跑还会因为文件
#   不存在直接崩——默认路径不该依赖任何未纳入版本控制/可能缺失的文件。
#   完整快照另有其人：训练启动时 save_config() 落盘的 runs/<tag>/config.json 才是那份
#   「实际生效的全量配置」，它由机器写、只读、用于复现，不参与上面的覆盖链。
#   优先级：dataclass 默认值 < CONFIG 指定的 exp/*.json < 命令行 --set
#
# 关于 seed：Config.seed 默认 null = 随机抽一个，但抽完会当场定死写进 config.json，
# 所以随手跑的也照样可复现。要跑 seed 组就显式给：--set seed=0，并把它写进 tag
# （…_s0），让目录名和实际 seed 对得上。
#
# 为什么 tmux 能扛住断线：tmux 服务端是脱离终端的独立进程，你的 ssh 连接只是
# 一个“客户端”。断线只是客户端没了，服务端和里面跑的 python 毫发无损。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs/_tmux"
PREFIX="cs336"
# 默认为空 = 不传 --config = 用 Config dataclass 里的默认值。
# 默认值的唯一事实来源是 python 代码，不是某个 JSON 文件；exp/*.json 是"实验",
# 要用哪个必须显式说出来。详见文件开头的“配置文件怎么组织”。
CONFIG="${CONFIG-}"

die() { echo "错误: $*" >&2; exit 1; }

# ---- 解释器：uv ---------------------------------------------------------
# 在“启动脚本的这个 shell”里把 uv 解析成绝对路径。tmux 会话的 PATH 未必和
# 你的交互式 shell 一样（尤其 uv 常装在 ~/.local/bin，靠 .bashrc 加进 PATH，
# 而非交互式 shell 不读 .bashrc）——固化绝对路径可以完全绕开这个坑。
UV="${UV:-$(command -v uv || true)}"
[[ -n "$UV" ]] || die "找不到 uv。若装在别处：UV=/path/to/uv $0 start train"
PY="$UV run python"
# 想跳过每次启动的依赖同步（比如 sweep 之间不改依赖），可以：
#   PY="$UV run --no-sync python"

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------ 作业表
# 加新实验族只改这里。第二列 = 这个作业认不认命令行参数（CONFIG / 透传）。
# 只写真实存在、且真的能接受这些参数的东西：脚本不认识的参数 python 那边
# 是静默忽略的，写多了等于给自己埋一个“以为生效了其实没生效”的雷。
CFG_ARG=""
[[ -n "$CONFIG" ]] && CFG_ARG="--config $CONFIG"

job_cmd() {
  case "$1" in
    train) echo "$PY -u -m cs336_basics.train${CFG_ARG:+ $CFG_ARG}" ;;
    lr)    echo "$PY -u -m cs336_basics.lr_sweep" ;;             # 网格 = lr_sweep.py 里的 LRS
    bs)    echo "$PY -u -m cs336_basics.batch_size_sweep" ;;
    *)     return 1 ;;
  esac
}

JOBS="train lr bs"
ARGV_JOBS="train"          # 这些作业才接受 CONFIG 和透传参数，其余传了直接报错

# ---- job[:tag] 解析 -----------------------------------------------------
# TAG 原样传给 python（可以带 / 分层）；KEY 是把 / 压成 - 之后的版本，只用来起
# tmux 会话名和日志文件名 —— 前者因为 tmux 的 target 语法是 session:window.pane，
# 后者因为文件名里不能有 /。
JOB=""; TAG=""; KEY=""
parse_job() {
  [[ -n "${1:-}" ]] || die "用法: $0 $ACTION <$(echo "$JOBS" | tr ' ' '|')>[:标签] [额外参数...]"
  JOB="${1%%:*}"
  TAG=""
  [[ "$1" == *:* ]] && TAG="${1#*:}"
  TAG="${TAG//[^A-Za-z0-9_\/-]/-}"   # 白名单：字母数字 _ - 和分层用的 /
  TAG="${TAG#/}"; TAG="${TAG%/}"     # 掐掉首尾的 /，免得拼出 runs//x
  KEY="$JOB${TAG:+-${TAG//\//-}}"
}

sess_of() { echo "${PREFIX}-$1"; }
log_of()  { echo "$LOG_DIR/$1.log"; }

# ------------------------------------------------------------------ start

cmd_start() {
  ACTION=start parse_job "${1:-}"; shift || true

  local base; base="$(job_cmd "$JOB")" || die "未知作业 '$JOB'（可选: $JOBS）"
  local sess; sess="$(sess_of "$KEY")"
  local log;  log="$(log_of "$KEY")"
  local spec="$JOB${TAG:+:$TAG}"

  # 不吃参数的作业，收到参数就当场拒绝。默默透传给一个没有 argparse 的脚本，
  # python 会连个警告都不给——等你发现扫的还是旧网格，一小时已经烧掉了。
  if [[ " $ARGV_JOBS " != *" $JOB "* ]]; then
    [[ $# -eq 0 ]] || die "$JOB 不接受命令行参数（收到: $*）。
       它的网格和基线配置写死在 python 文件里，改实验请直接改那里。"
    [[ -z "$CONFIG" ]] || die "$JOB 不读 CONFIG（你设了 CONFIG=$CONFIG，它不会生效）。
       基线配置写死在 sweep 脚本顶部的 BASE dict，要换请改那一行。"
  fi

  # 起飞前检查配置文件。不查的话：tmux 会话照样建起来，python 一秒后就 FileNotFound
  # 死掉，你还得 log 一遍才知道原因——错误离犯错的地方越近越好。
  if [[ "$base" == *--config* && ! -f "$PROJECT_DIR/$CONFIG" ]]; then
    die "找不到配置文件 $CONFIG（相对 $PROJECT_DIR）。三选一：
       换一个存在的  : CONFIG=exp/small.json $0 start $spec
       先照默认值生成: $0 init-config $CONFIG   （生成后删掉与默认值相同的项）
       干脆不用文件  : $0 start $spec           （默认就是走 dataclass 默认值）"
  fi

  if tmux has-session -t "=$sess" 2>/dev/null; then
    die "会话 $sess 已存在。先 '$0 stop $KEY'，或 '$0 attach $KEY' 看看它在干嘛。"
  fi

  # 额外参数原样追加（如 --set batch_size=64）。用 %q 逐个转义而不是 "$*"：
  # 后者把参数拍平成一个字符串，--set 'note=a b' 会散成三个参数。
  local full="$base"
  if [[ $# -gt 0 ]]; then full="$base $(printf '%q ' "$@")"; full="${full% }"; fi

  # 把冒号后面那截交给 Config.tag —— 于是会话名、本脚本日志、runs/ 下的产物目录
  # 三者同名。sweep 类作业给每个 run 自己命名，不能这么覆盖，所以只对 train 做；
  # 你自己已经 --set tag=... 了就不插手。
  if [[ "$JOB" == train && -n "$TAG" && "$full" != *" tag="* ]]; then
    if [[ "$full" == *" --set "* ]]; then
      full="$full tag=$TAG"
    else
      full="$full --set tag=$TAG"
    fi
  fi

  # 把要跑的东西写成一个 runner 脚本，而不是塞进 tmux 命令行 —— 省掉多层引号转义，
  # 出问题时还能直接 bash 这个文件复现。
  local runner="$LOG_DIR/$KEY.run.sh"
  # 日志开头那行「$ 实际命令」用单引号原样打印：直接 echo 的话 bash 会把
  # --set 'name=a b' 里的引号吃掉，回头看日志会以为传的是两个参数。
  local disp=${full//\'/\'\\\'\'}
  {
    echo '#!/usr/bin/env bash'
    echo "cd \"$PROJECT_DIR\"                # uv 靠 cwd 找 pyproject.toml 定位项目环境"
    [[ -n "${GPU:-}" ]] && echo "export CUDA_VISIBLE_DEVICES=$GPU"
    cat <<EOF
exec > >(tee -a "$log") 2>&1          # 屏幕和日志各留一份
echo "=== $sess 开始 \$(date -Is) ==="
printf '\$ %s\n' '$disp'
echo
$full
rc=\$?
echo
echo "=== $sess 结束 rc=\$rc \$(date -Is) ==="
echo "(窗口保留着方便回看；'$0 stop $KEY' 关闭)"
sleep infinity                        # 别让 pane 立刻消失，否则报错信息一闪而过
EOF
  } > "$runner"
  chmod +x "$runner"

  : > "$log"                          # 新一轮，日志清空（历史想留就改成 mv 备份）
  tmux new-session -d -s "$sess" -c "$PROJECT_DIR" bash "$runner"

  echo "已在 tmux 会话 $sess 中启动："
  [[ -n "${GPU:-}" ]] && echo "  CUDA_VISIBLE_DEVICES=$GPU"
  echo "  $full"
  if [[ "$JOB" == train ]]; then
    if [[ -n "$TAG" ]]; then echo "  产物  : runs/$TAG/"
    else echo "  产物  : runs/_scratch/<按超参自动拼的名>/   （没给标签，视为随手跑）"; fi
  fi
  echo
  echo "  进去看   : $0 attach $KEY     (Ctrl-b 再按 d 脱离)"
  echo "  追日志   : $0 log $KEY        (Ctrl-C 只退出 tail)"
  echo "  看状态   : $0 ls"
  echo "  停止     : $0 stop $KEY"
  echo
  echo "现在可以直接关掉这个终端 / 断开 ssh。"
}

# ------------------------------------------------------------------ 查看

# 判断是否还在跑：看 pane 里那个 runner bash 的直接子进程。
# 跑的时候是 tee + uv（uv 下面才是真正的 python）；结束后只剩 tee + sleep。
# 所以“除 tee/sleep 外还有别的子进程”就等于还在跑——
# 不依赖进程名到底叫 python 还是 uv，换解释器也不用改这里。
#
# 注：不能用 #{pane_current_command}，它一直报 bash（runner 脚本本身），
# 子进程换来换去它都不变，区分不出运行与结束。
running_of() {
  local pid c comm
  pid="$(tmux list-panes -t "$1" -F '#{pane_pid}' 2>/dev/null | head -n1)"
  [[ -n "$pid" ]] || return 1
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    comm="$(ps -o comm= -p "$c" 2>/dev/null | tr -d ' ')"
    case "$comm" in
      tee|sleep|"") continue ;;
      *) return 0 ;;
    esac
  done
  return 1
}

cmd_ls() {
  if ! tmux has-session 2>/dev/null; then echo "没有任何 tmux 会话"; return; fi
  # 表头用 %-24s：printf 按字节补齐，而“会话”两个汉字占 6 字节、只显示 2 格宽，
  # 多给 2 个字节才和下面纯 ASCII 的会话名对齐。
  printf "%-24s %-8s %s\n" "会话" "状态" "日志最后一行"
  printf -- "----------------------------------------------------------------------\n"
  tmux list-sessions -F '#{session_name}' | while read -r sess; do
    [[ "$sess" == ${PREFIX}-* ]] || continue
    key="${sess#${PREFIX}-}"
    # 注意别写成 `running_of "$sess" && state=运行中`：作业跑完时它返回非 0，
    # 作为独立语句会被 set -e 当成失败直接退出整个脚本。
    if running_of "$sess"; then state="运行中"; else state="已结束"; fi
    last=""
    [[ -f "$(log_of "$key")" ]] && last="$(tail -n 1 "$(log_of "$key")" | cut -c1-60)"
    printf "%-22s %-8s %s\n" "$sess" "$state" "$last"
  done
}

cmd_attach() {
  ACTION=attach parse_job "${1:-}"
  local sess; sess="$(sess_of "$KEY")"
  tmux has-session -t "=$sess" 2>/dev/null || die "会话 $sess 不存在"
  echo "提示：脱离用 Ctrl-b 然后 d。按 Ctrl-C 会打断训练！"
  sleep 1
  tmux attach -t "=$sess"
}

cmd_log() {
  ACTION=log parse_job "${1:-}"
  local log; log="$(log_of "$KEY")"
  [[ -f "$log" ]] || die "还没有日志 $log"
  tail -f -n 40 "$log"      # 只读，Ctrl-C 安全
}

# ------------------------------------------------------------------ 配置文件

# 从 Config dataclass 导出一份默认配置。单一事实来源仍是 dataclass，
# JSON 只是它的快照——手写 JSON 迟早和 dataclass 漂移。
cmd_init_config() {
  local out="${1:-${CONFIG:-exp/base.json}}"
  [[ -e "$PROJECT_DIR/$out" ]] && die "$out 已存在，不覆盖。想重来先手动删掉。"

  ( cd "$PROJECT_DIR" && $PY - "$out" <<'PYEOF'
import importlib, json, sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

Config = None
for mod in ("cs336_basics.data", "cs336_basics.config", "cs336_basics.train"):
    try:
        obj = getattr(importlib.import_module(mod), "Config", None)
        if is_dataclass(obj):
            Config = obj
            break
    except Exception:
        continue
if Config is None:
    sys.exit("找不到 Config dataclass；把它所在的模块名加进本脚本 init-config 的搜索列表")

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(asdict(Config()), indent=2, ensure_ascii=False) + "\n")
print(f"已写出 {out}（来自 {Config.__module__}.Config 的默认值）")
PYEOF
  )
  echo
  echo "这是一份「全量默认值」快照，不是给你直接用的——请把和默认值相同的行删掉，"
  echo "只留这个实验真正要改的几项。配置文件是 diff，不是副本；留成副本以后改"
  echo "dataclass 默认值就再也传播不到它了。"
  echo "另：JSON 没有元组，betas 这类会存成 [0.9, 0.95]，载回来是 list。"
  echo "另：seed 会是 null（= 每次随机），tag 会是空串（= 落到 _scratch）；"
  echo "    这两项通常留给命令行给，别写死在 exp/*.json 里。"
}

# ------------------------------------------------------------------ 停止 / 清理

cmd_stop() {
  ACTION=stop parse_job "${1:-}"
  local sess; sess="$(sess_of "$KEY")"
  tmux has-session -t "=$sess" 2>/dev/null || { echo "会话 $sess 不存在"; return; }
  tmux kill-session -t "=$sess"     # 杀会话 = 杀掉 pane 里的整棵进程树
  echo "已停止 $sess（日志保留在 $(log_of "$KEY")）"
  sleep 1
  if pgrep -f "[c]s336_basics" >/dev/null 2>&1; then
    echo "⚠ 还有 cs336 进程残留，检查: pgrep -fa '[c]s336_basics'"
  fi
}

cmd_clean() {
  tmux has-session 2>/dev/null || { echo "没有会话可清理"; return; }
  tmux list-sessions -F '#{session_name}' | while read -r sess; do
    [[ "$sess" == ${PREFIX}-* ]] || continue
    if ! running_of "$sess"; then
      tmux kill-session -t "=$sess"
      echo "清理已结束的会话 $sess"
    fi
  done
}

# ------------------------------------------------------------------ 入口

case "${1:-}" in
  start)  shift; cmd_start "$@" ;;
  ls|list|status) cmd_ls ;;
  attach|a) shift; cmd_attach "$@" ;;
  log|tail) shift; cmd_log "$@" ;;
  stop|kill) shift; cmd_stop "$@" ;;
  clean)  cmd_clean ;;
  init-config) shift; cmd_init_config "$@" ;;
  *)
    # 打印文件开头的注释块（第 2 行起，遇到第一个非注释行停）
    awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}"
    echo
    echo "可用作业: $JOBS"
    echo "配置文件: $CONFIG"
    echo "解释器  : $PY"
    ;;
esac