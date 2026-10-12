#!/usr/bin/env bash
# 监控 GitHub PR vllm-project/vllm-ascend #17834 的状态变化
# 每 30 分钟拉一次，有实质变化才输出
set -u
TOKEN="${GH_TOKEN:?set GH_TOKEN env var}"
OWNER="vllm-project"
REPO="vllm-ascend"
PR=17834
SHA="36de58a9097647d530314e0c2fd9934485a39144"
API="https://api.github.com/repos/$OWNER/$REPO"
INTERVAL=1800  # 30 min
STATE_FILE="/tmp/pr17834_state.txt"

touch "$STATE_FILE"

while true; do
  # labels
  labels=$(curl -sS -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
    "$API/issues/$PR/labels" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join(l['name'] for l in d) if d else 'none')" 2>/dev/null)

  # PR merged?
  merged=$(curl -sS -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
    "$API/pulls/$PR" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('MERGED' if d.get('merged') else 'open')" 2>/dev/null)

  # check-runs
  checks=$(curl -sS -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
    "$API/commits/$SHA/check-runs?per_page=100" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('|'.join(f\"{r['name']}={r['conclusion'] or r['status']}\" for r in d.get('check_runs',[])))" 2>/dev/null)

  # comments count + last author
  comments=$(curl -sS -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
    "$API/issues/$PR/comments" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{len(d)} last={d[-1]['user']['login']}@{d[-1]['created_at']}\" if d else '0')" 2>/dev/null)

  # reviews
  reviews=$(curl -sS -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
    "$API/pulls/$PR/reviews" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{len(d)} last={d[-1]['user']['login']}:{d[-1]['state']}@{d[-1]['submitted_at']}\" if d else '0')" 2>/dev/null)

  current="merged=$merged|labels=$labels|comments=$comments|reviews=$reviews|checks=$checks"

  if [ "$current" != "$(cat "$STATE_FILE" 2>/dev/null)" ]; then
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') PR #$PR 状态变化 ====="
    echo "$current"
    echo ""
    echo "$current" > "$STATE_FILE"
  fi

  sleep "$INTERVAL"
done
