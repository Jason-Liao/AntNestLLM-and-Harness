# -*- coding: utf-8 -*-
"""Agent 25 + Agent 20：在线自我进化（M5-4，"使用即训练"自动化）。

闭环：新轨迹检测 → SFT 增量续训（含轨迹回流）→ 独立评测 → 择优晋升。
每次进化追加一行到 artifacts/evolve_log.jsonl，形成可审计的进化史。

用法：
  python -m antnest_llm.evolve                 # 检测新轨迹并进化（无新轨迹则跳过）
  python -m antnest_llm.evolve --force         # 强制跑一轮增量训练
  python -m antnest_llm.evolve --status        # 查看进化史
产出：artifacts/{evo{N}_ckpt.pt, ...}，最优者被 AntNestLLMClient 自动发现
     （前缀 evo 不在 rank 规则内时按预训练处理，见 --promote 逻辑）。
"""
import argparse
import json
import time
from pathlib import Path

ART = Path(__file__).resolve().parent.parent / "artifacts"
TRAJ = ART / "trajs.jsonl"
LOG = ART / "evolve_log.jsonl"


def read_log() -> list:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def new_traj_count(last_seen: int) -> int:
    """统计 trajs.jsonl 中 last_seen 之后新增的成功轨迹数。"""
    if not TRAJ.exists():
        return 0
    n = 0
    for line in TRAJ.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if t.get("ok"):
            n += 1
    return max(0, n - last_seen)


def best_ckpt() -> str:
    """当前最优产物前缀（与 AntNestLLMClient 相同的 rank 规则）。"""
    rank = lambda p: 2 if p.startswith("grpo") else (1 if p.startswith("sft") else 0)
    cands = []
    for cf in ART.glob("*model_config.json"):
        pfx = cf.name[: -len("model_config.json")].rstrip("_")
        if pfx.startswith("evo"):
            continue  # 进化产物不作为进化起点，避免链式漂移
        ck = ART / f"{pfx}_ckpt.pt"
        if ck.exists():
            cands.append((rank(pfx), cf.stat().st_mtime, pfx))
    return max(cands)[2] if cands else ""


def run_evolve(force=False, sft_steps=60) -> dict:
    hist = read_log()
    last_seen = hist[-1]["total_trajs"] if hist else 0
    n_new = new_traj_count(last_seen)
    if n_new == 0 and not force:
        return {"action": "skip", "reason": "无新增成功轨迹", "total_trajs": last_seen}

    base = best_ckpt()
    if not base:
        return {"action": "skip", "reason": "无可用基座 checkpoint"}

    gen = len(hist) + 1
    out = f"evo{gen}"
    t0 = time.time()
    import subprocess, sys
    # SFT 增量：以当前最优为基座，短程续训（含轨迹回流）
    r = subprocess.run(
        [sys.executable, "-m", "antnest_llm.sft",
         "--steps", str(sft_steps), "--base_prefix", base,
         "--out_prefix", out, "--traj", str(TRAJ)],
        cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
    if r.returncode != 0:
        return {"action": "fail", "stage": "sft", "error": r.stderr[-400:]}

    # 独立评测择优：新产物 vs 基座
    from .eval import evaluate
    old_r = evaluate(base, verbose=False)
    new_r = evaluate(out, verbose=False)
    better = new_r["overall"] >= old_r["overall"]
    rec = {
        "action": "evolve", "gen": gen, "base": base, "out": out,
        "n_new_trajs": n_new, "promoted": better,
        "old_overall": old_r["overall"], "new_overall": new_r["overall"],
        "total_trajs": last_seen + n_new,
        "seconds": round(time.time() - t0, 1), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="无新轨迹也强制进化一轮")
    ap.add_argument("--status", action="store_true", help="仅打印进化史")
    ap.add_argument("--sft_steps", type=int, default=60, help="增量 SFT epoch 数")
    args = ap.parse_args()

    if args.status:
        hist = read_log()
        print(f"进化史：{len(hist)} 轮")
        for h in hist:
            print(f"  gen{h.get('gen')} | {h.get('base')}→{h.get('out')} | "
                  f"overall {h.get('old_overall')}→{h.get('new_overall')} | "
                  f"promoted={h.get('promoted')} | +{h.get('n_new_trajs')}轨迹")
        return

    rec = run_evolve(force=args.force, sft_steps=args.sft_steps)
    if rec["action"] == "skip":
        print(f"[跳过] {rec['reason']}（成功轨迹共 {rec.get('total_trajs', 0)} 条）")
    elif rec["action"] == "fail":
        print(f"[失败] {rec['stage']}: {rec['error']}")
    else:
        flag = "✓ 晋升" if rec["promoted"] else "✗ 保留原模型"
        print(f"[gen{rec['gen']}] {rec['base']} → {rec['out']} | "
              f"overall {rec['old_overall']} → {rec['new_overall']} | {flag} | "
              f"+{rec['n_new_trajs']} 轨迹 | {rec['seconds']}s")


if __name__ == "__main__":
    main()
