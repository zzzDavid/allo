#!/bin/bash
# Stage harness for the release milestone scripts (docs/release/VALIDATION.md,
# round 107).  Every stage runs under `stage`, which keeps the complete output
# of the stage in its own log, records the command's own exit status (not the
# status of a `tail` or `grep` behind it), and marks the run failed when a
# required stage fails.  `milestone_finish` prints the summary and returns
# nonzero if any required stage failed, so a wrapper's `DONE` cannot hide a
# failure.  Usage:
#   source stage_lib.sh; milestone_begin "$STAGE_DIR"
#   stage NAME -- cmd args...            # required stage
#   stage --optional NAME -- cmd args... # recorded, does not fail the run
#   milestone_finish                     # exit status of the whole run
set -u
MILESTONE_FAILED=0
MILESTONE_SUMMARY=""

milestone_begin() {
  STAGE_DIR=$1
  mkdir -p "$STAGE_DIR"
  : > "$STAGE_DIR/SUMMARY.txt"
  echo "milestone begin $(date -Is) host $(hostname) stage_dir $STAGE_DIR" | tee -a "$STAGE_DIR/SUMMARY.txt"
}

stage() {
  local required=1
  if [ "$1" = "--optional" ]; then required=0; shift; fi
  local name=$1; shift
  [ "$1" = "--" ] && shift
  local log="$STAGE_DIR/$name.log"
  echo "=== stage $name begin $(date -Is)" | tee -a "$STAGE_DIR/SUMMARY.txt"
  echo "command: $*" > "$log"
  local rc
  # The stage's own status: `set -o pipefail` plus PIPESTATUS[0] of the command,
  # independent of what tee returns.
  ( set -o pipefail; "$@" 2>&1 | tee -a "$log" > /dev/null ); rc=$?
  local verdict=PASS
  if [ $rc -ne 0 ]; then
    verdict=FAIL
    if [ $required -eq 1 ]; then MILESTONE_FAILED=1; fi
  fi
  echo "$verdict rc=$rc required=$required $(date -Is)" > "$STAGE_DIR/$name.status"
  echo "=== stage $name $verdict rc=$rc required=$required (log: $log)" | tee -a "$STAGE_DIR/SUMMARY.txt"
  # The tail of the log is printed for the reader; the whole log stays on disk.
  tail -n 4 "$log" | sed 's/^/    /'
  return $rc
}

milestone_finish() {
  echo "milestone end $(date -Is)" | tee -a "$STAGE_DIR/SUMMARY.txt"
  if [ $MILESTONE_FAILED -ne 0 ]; then
    echo "MILESTONE FAILED (a required stage failed; see $STAGE_DIR/*.status)" | tee -a "$STAGE_DIR/SUMMARY.txt"
    return 1
  fi
  echo "MILESTONE PASSED (every required stage exited 0)" | tee -a "$STAGE_DIR/SUMMARY.txt"
  return 0
}
