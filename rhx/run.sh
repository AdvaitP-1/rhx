#!/usr/bin/env bash
# run.sh -- one-command entry point for the RHX experiment program.
#
#   ./run.sh check                 verify the machine and the build
#   ./run.sh selftest              36 correctness checks, no root needed
#   ./run.sh gate0 [OUT]           can we measure f(lambda)?
#   ./run.sh gate1 [OUT]           pre-registered prediction test, 20 trials
#   ./run.sh gate2 [OUT]           homogeneous negative control, 20 trials
#   ./run.sh all [OUT]             gate0 -> gate1 -> gate2, stopping on failure
#
# Gates are ordered so each can falsify the theory before the next runs.
# Everything after "check" that touches cgroups needs root.

set -euo pipefail
cd "$(dirname "$0")"

BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf "%s\n" "$*"; }
head_() { printf "\n%s%s%s\n" "$BOLD" "$*" "$OFF"; }
ok()   { printf "%s  ok%s   %s\n" "$GRN" "$OFF" "$*"; }
warn() { printf "%s  warn%s %s\n" "$YEL" "$OFF" "$*"; }
bad()  { printf "%s  FAIL%s %s\n" "$RED" "$OFF" "$*"; }

PY="${PYTHON:-python3}"
PAGES="${PAGES:-200000}"        # 200k pages = 800 MiB
MEASURE_S="${MEASURE_S:-300}"
WINDOW_T="${WINDOW_T:-60}"
TRIALS="${TRIALS:-20}"
MODE="${MODE:-proactive}"       # proactive | natural
EXTRA_ARGS=()
if [ -n "${EXTRA:-}" ]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA"
fi

build() {
  head_ "build"
  ( cd workload && make -s ) && ok "rategen built"
}

check() {
  head_ "preflight"
  local fail=0

  [ "$(uname -s)" = "Linux" ] && ok "Linux host" || { bad "not Linux -- no cgroups, no PSI, no DAMON"; fail=1; }

  if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    grep -q memory /sys/fs/cgroup/cgroup.controllers \
      && ok "cgroup v2 with memory controller" \
      || { bad "cgroup v2 present but memory controller not delegated"; fail=1; }
  else bad "cgroup v2 unified hierarchy not mounted"; fail=1; fi

  [ "$(id -u)" = "0" ] && ok "running as root" || { bad "not root -- cannot create cgroups or write memory.reclaim"; fail=1; }

  if command -v systemd-detect-virt >/dev/null 2>&1; then
    local v; v="$(systemd-detect-virt || true)"
    [ "$v" = "none" ] && ok "bare metal" || { bad "virtualized ($v) -- development only, not publishable"; fail=1; }
  else warn "cannot detect virtualization"; fi

  if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
    if grep -q '\[always\]' /sys/kernel/mm/transparent_hugepage/enabled; then
      bad "THP=always blurs per-4KiB residency. Fix: echo madvise > /sys/kernel/mm/transparent_hugepage/enabled"
      fail=1
    else ok "THP is not 'always'"; fi
  fi

  [ -f /sys/kernel/mm/lru_gen/enabled ] \
    && ok "MGLRU state readable ($(cat /sys/kernel/mm/lru_gen/enabled))" \
    || warn "MGLRU state unreadable -- victim policy will be unrecorded"

  [ -d /sys/kernel/mm/damon/admin ] && ok "DAMON sysfs present" || warn "DAMON absent (workload self-report still works)"
  [ -f /proc/pressure/memory ] && ok "PSI available" || { bad "PSI absent -- rebuild kernel with CONFIG_PSI"; fail=1; }

  "$PY" -c 'import numpy,scipy' 2>/dev/null && ok "numpy + scipy" || { bad "pip install numpy scipy"; fail=1; }
  [ -x workload/rategen ] && ok "rategen built" || { warn "rategen not built -- run: ./run.sh build"; }

  echo
  if [ "$fail" = "0" ]; then
    say "${GRN}${BOLD}READY${OFF} -- results from this machine are publishable."
  else
    say "${RED}${BOLD}NOT READY${OFF} -- fix the FAIL lines above."
    say "To proceed anyway for development, add --allow-unpublishable to the gate command."
  fi
  return "$fail"
}

gate0() {
  local out="${1:-runs/gate0}"; build
  head_ "Gate 0 -- can we measure f(lambda)?  (${TRIALS} trials)"
  "$PY" run_trials.py --gate 0 --trials "$TRIALS" --out "$out" \
      --pages "$PAGES" --measure-s "$MEASURE_S" --window-T "$WINDOW_T" \
      --dist gamma --p1 0.8 --p2 0.5 \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

gate1() {
  local out="${1:-runs/gate1}"; build
  head_ "Gate 1 -- pre-registered prediction test  (${TRIALS} trials, mode=${MODE})"
  "$PY" run_trials.py --gate 1 --trials "$TRIALS" --out "$out" \
      --reclaim-mode "$MODE" --pages "$PAGES" --measure-s "$MEASURE_S" \
      --window-T "$WINDOW_T" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

gate2() {
  local out="${1:-runs/gate2}"; build
  head_ "Gate 2 -- homogeneous negative control  (${TRIALS} trials, mode=${MODE})"
  "$PY" run_trials.py --gate 2 --trials "$TRIALS" --out "$out" \
      --reclaim-mode "$MODE" --pages "$PAGES" --measure-s "$MEASURE_S" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

case "${1:-help}" in
  build)    build ;;
  check)    check ;;
  selftest) build; "$PY" selftest.py ;;
  gate0)    gate0 "${2:-runs/gate0}" ;;
  gate1)    gate1 "${2:-runs/gate1}" ;;
  gate2)    gate2 "${2:-runs/gate2}" ;;
  all)
    OUT="${2:-runs}"
    build; check || { say "preflight failed; aborting"; exit 1; }
    gate0 "$OUT/gate0" || { say "${RED}Gate 0 failed. Gate 1 would be uninterpretable: a prediction miss could not be attributed to the theory rather than the estimator. Stopping.${OFF}"; exit 1; }
    gate1 "$OUT/gate1" || { say "${RED}Gate 1 failed. See the diagnosis: offset-like implicates the estimator, shape-like falsifies the theory. Stopping.${OFF}"; exit 1; }
    gate2 "$OUT/gate2" || { say "${RED}Gate 2 failed.${OFF}"; exit 1; }
    say "\n${GRN}${BOLD}All gates passed.${OFF} Gate 3 (real workloads) is deliberately not implemented."
    ;;
  *)
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
