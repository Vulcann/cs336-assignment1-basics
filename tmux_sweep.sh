#!/usr/bin/env bash
# tmux_sweep.sh —— 用 tmux 把 sweep 挂到后台，关掉 ssh / 终端也照跑。（uv 版）
#
#   ./tmux_sweep.sh start lr            # 起 lr sweep（tmux 会话 cs336-lr）
#   ./tmux_sweep.sh start bs            # 起 batch_size sweep
#   ./tmux_sweep.sh start lr --force    # 多余的参数原样透传给 python 脚本
#   ./tmux_sweep.sh ls                  # 看所有会话 + 运行/已完成
#   ./tmux_sweep.sh attach lr           # 进去看（Ctrl-b 然后 d 脱离，别按 Ctrl-C）
#   ./tmux_sweep.sh log lr              # 在当前终端 tail -f 日志（Ctrl-C 只退 tail）
#   ./tmux_sweep.sh stop lr             # 停掉这个会话（连带杀掉里面的 python）
#   ./tmux_sweep.sh clean               # 清掉已结束的会话
#
# 为什么 tmux 能扛住断线：tmux 服务端是脱离终端的独立进程，你的 ssh 连接只是
# 一个“客户端”。断线只是客户端没了，服务端和里面跑的 python 毫发无损。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs/_tmux"
PREFIX="cs336"

die() { echo "错误: $*" >&2; exit 1; }

# ---- 解释器：uv ---------------------------------------------------------
# 在“启动脚本的这个 shell”里把 uv 解析成绝对路径。tmux 会话的 PATH 未必和
# 你的交互式 shell 一样（尤其 uv 常装在 ~/.local/bin，靠 .bashrc 加进 PATH，
# 而非交互式 shell 不读 .bashrc）——固化绝对路径可以完全绕开这个坑。
UV="${UV:-$(command -v uv || true)}"
[[ -n "$UV" ]] || die "找不到 uv。若装在别处：UV=/path/to/uv $0 start lr"
PY="$UV run python"
# 想跳过每次启动的依赖同步（比如 sweep 之间不改依赖），可以：
#   PY="$UV run --no-sync python"

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------ 作业表
# 加新实验族只改这里。
job_cmd() {
  case "$1" in
    lr)     echo "$PY -u -m cs336_basics.sweep --grid lr --config exp/base.json" ;;
    bs)     echo "$PY -u -m cs336_basics.batch_size_sweep" ;;
    depth)  echo "$PY -u -m cs336_basics.sweep --grid depth --config exp/base.json" ;;
    *)      return 1 ;;
  esac
}

JOBS="lr bs depth"

sess_of() { echo "${PREFIX}-$1"; }
log_of()  { echo "$LOG_DIR/$1.log"; }

# ------------------------------------------------------------------ start

cmd_start() {
  local job="${1:-}"; shift || true
  [[ -n "$job" ]] || die "用法: $0 start <$(echo "$JOBS" | tr ' ' '|')> [额外参数...]"

  local base; base="$(job_cmd "$job")" || die "未知作业 '$job'（可选: $JOBS）"
  local sess; sess="$(sess_of "$job")"
  local log;  log="$(log_of "$job")"

  if tmux has-session -t "=$sess" 2>/dev/null; then
    die "会话 $sess 已存在。先 '$0 stop $job'，或 '$0 attach $job' 看看它在干嘛。"
  fi

  # 额外参数原样追加（如 --force / --devices 0,1）
  local full="$base"
  if [[ $# -gt 0 ]]; then full="$base $*"; fi

  # 把要跑的东西写成一个 runner 脚本，而不是塞进 tmux 命令行 —— 省掉多层引号转义，
  # 出问题时还能直接 bash 这个文件复现。
  local runner="$LOG_DIR/$job.run.sh"
  cat > "$runner" <<EOF
#!/usr/bin/env bash
cd "$PROJECT_DIR"                     # uv 靠 cwd 找 pyproject.toml 定位项目环境
exec > >(tee -a "$log") 2>&1          # 屏幕和日志各留一份
echo "=== $sess 开始 \$(date -Is) ==="
echo "\$ $full"
echo
$full
rc=\$?
echo
echo "=== $sess 结束 rc=\$rc \$(date -Is) ==="
echo "(窗口保留着方便回看；'$0 stop $job' 关闭)"
sleep infinity                        # 别让 pane 立刻消失，否则报错信息一闪而过
EOF
  chmod +x "$runner"

  : > "$log"                          # 新一轮，日志清空（历史想留就改成 mv 备份）
  tmux new-session -d -s "$sess" -c "$PROJECT_DIR" bash "$runner"

  echo "已在 tmux 会话 $sess 中启动："
  echo "  $full"
  echo
  echo "  进去看   : $0 attach $job     (Ctrl-b 再按 d 脱离)"
  echo "  追日志   : $0 log $job        (Ctrl-C 只退出 tail)"
  echo "  看状态   : $0 ls"
  echo "  停止     : $0 stop $job"
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
  printf "%-16s %-8s %s\n" "会话" "状态" "日志最后一行"
  printf -- "------------------------------------------------------------------\n"
  tmux list-sessions -F '#{session_name}' | while read -r sess; do
    [[ "$sess" == ${PREFIX}-* ]] || continue
    local job="${sess#${PREFIX}-}"
    local state="已结束"
    running_of "$sess" && state="运行中"
    local last=""
    [[ -f "$(log_of "$job")" ]] && last="$(tail -n 1 "$(log_of "$job")" | cut -c1-70)"
    printf "%-16s %-8s %s\n" "$sess" "$state" "$last"
  done
}

cmd_attach() {
  local job="${1:?用法: $0 attach <job>}"
  local sess; sess="$(sess_of "$job")"
  tmux has-session -t "=$sess" 2>/dev/null || die "会话 $sess 不存在"
  echo "提示：脱离用 Ctrl-b 然后 d。按 Ctrl-C 会打断训练！"
  sleep 1
  tmux attach -t "=$sess"
}

cmd_log() {
  local job="${1:?用法: $0 log <job>}"
  local log; log="$(log_of "$job")"
  [[ -f "$log" ]] || die "还没有日志 $log"
  tail -f -n 40 "$log"      # 只读，Ctrl-C 安全
}

# ------------------------------------------------------------------ 停止 / 清理

cmd_stop() {
  local job="${1:?用法: $0 stop <job>}"
  local sess; sess="$(sess_of "$job")"
  tmux has-session -t "=$sess" 2>/dev/null || { echo "会话 $sess 不存在"; return; }
  tmux kill-session -t "=$sess"     # 杀会话 = 杀掉 pane 里的整棵进程树
  echo "已停止 $sess（日志保留在 $(log_of "$job")）"
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
  *)
    # 打印文件开头的注释块（第 2 行起，遇到第一个非注释行停）
    awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}"
    echo
    echo "可用作业: $JOBS"
    echo "解释器  : $PY"
    ;;
esac
