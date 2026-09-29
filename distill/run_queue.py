"""Run a list of train_inject_lrsd.py (or train_lrsd.py) jobs with up to --max_parallel concurrent processes per GPU.

jobs file: JSON list of {"name": str, "args": [str, ...]}; each job writes to
<runs_dir>/<name>/ (log.jsonl, stdout.log).  A job is skipped when its log already contains the
final epoch (train_inject_lrsd.py) or step (train_lrsd.py), so the queue can be re-launched after an interruption.

Scheduling: every pass (5-15 s) each GPU with a free slot takes the next job, emptiest GPU first, so all GPUs
get work within seconds.  After a launch the SAME GPU waits --settle seconds before its next launch (its
memory shows up late, and --min_free_gb is checked on it); other GPUs are not held up.
Re-launching while runs of an earlier queue are still alive adopts them (found through /proc by their
--out_dir; they keep their slot on their GPU and are not started twice).

Usage (--script is looked up in distill/ when it is not found from the current directory):
  python run_queue.py --jobs jobs_inject_scale.json --runs_dir ../experiment_logs/distill/inject --gpus 0,1 \\
      --max_parallel 2 [--dry_run: print what would be skipped / adopted / started, then exit]
"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def done(out_dir, steps, epochs=None):
    p = os.path.join(out_dir, "log.jsonl")
    if not os.path.exists(p):
        return False
    try:
        last = [json.loads(x) for x in open(p) if x.strip()][-1]
        if epochs is not None:
            return last.get("epoch") == epochs
        return last.get("step") == steps and "held_uq_em" in last
    except Exception:
        return False


def gpu_gb(g, field="memory.free"):
    try:
        out = subprocess.check_output(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits",
                                       "-i", str(g)]).decode().strip()
        return float(out) / 1024
    except Exception:
        return 0.0


def free_gb(g):
    return gpu_gb(g)


def running(script, outs):
    """{out_dir: (pid, gpu)} for live processes of `script` started with one of these --out_dir."""
    found = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = [a.decode() for a in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if a]
            if "--out_dir" not in argv or not any(os.path.basename(a) == os.path.basename(script) for a in argv):
                continue
            out = os.path.normpath(os.path.join(os.readlink(f"/proc/{pid}/cwd"), argv[argv.index("--out_dir") + 1]))
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            gpu = next((e.decode().split("=", 1)[1] for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
        except (OSError, ValueError, IndexError):
            continue
        if out in outs and Adopted(int(pid)).poll() is None:
            found[out] = (int(pid), gpu)
    return found


class Adopted:
    """A run started by an earlier queue: not our child, so only liveness is known (no exit code)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        try:
            state = open(f"/proc/{self.pid}/stat").read().rsplit(")", 1)[1].split()[0]
            if state != "Z":
                return None
        except (OSError, IndexError):
            pass
        self.returncode = "?"
        return self.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    ap.add_argument("--max_parallel", type=int, default=4, help="per GPU")
    ap.add_argument("--script", default="train_inject_lrsd.py", help="train_inject_lrsd.py or train_lrsd.py")
    ap.add_argument("--min_free_gb", type=float, default=0.0, help="only start a job on a GPU with this much free memory")
    ap.add_argument("--settle", type=float, default=None,
                    help="seconds between two launches on the same GPU (default 60 with --min_free_gb, else 20)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    settle = args.settle if args.settle is not None else (60 if args.min_free_gb > 0 else 20)
    if not os.path.exists(args.script) and os.path.exists(os.path.join(HERE, args.script)):
        args.script = os.path.join(HERE, args.script)
    jobs = json.load(open(args.jobs))
    if os.path.basename(args.script) != "train_lrsd.py" and any("--steps" in j["args"] for j in jobs):
        sys.exit(f"{args.jobs} holds train_lrsd.py jobs (--steps); pass --script train_lrsd.py")
    gpus = [g for g in args.gpus.split(",")]
    slots = {g: [] for g in gpus}
    last = {g: 0.0 for g in gpus}
    todo = []
    for j in jobs:
        out = os.path.join(args.runs_dir, j["name"])
        steps = int(j["args"][j["args"].index("--steps") + 1]) if "--steps" in j["args"] else 300
        epochs = None
        if args.script != "train_lrsd.py":
            epochs = int(j["args"][j["args"].index("--epochs") + 1]) if "--epochs" in j["args"] else 50
        if done(out, steps, epochs):
            print("skip (done)", j["name"], flush=True)
            continue
        todo.append((j, out))
    alive = running(args.script, {os.path.abspath(out) for _, out in todo})
    pending = []
    for j, out in todo:
        if os.path.abspath(out) in alive:
            pid, g = alive[os.path.abspath(out)]
            print(f"adopt (running, pid {pid}, gpu {g})", j["name"], flush=True)
            if g in slots:
                slots[g].append((j["name"], Adopted(pid)))
                last[g] = time.time()
        else:
            pending.append((j, out))
    print(f"{len(pending)} jobs pending", flush=True)
    if args.dry_run:
        for j, _ in pending[:10]:
            print("  next:", j["name"])
        return
    if args.min_free_gb > 0:
        for g in gpus:
            total = gpu_gb(g, "memory.total")
            if total < args.min_free_gb:
                sys.exit(f"gpu {g}: {'nvidia-smi cannot be queried' if total == 0 else f'only {total:.0f} GB in total'}, "
                         f"so no job would ever start with --min_free_gb {args.min_free_gb:g}; lower it (e.g. 20 for "
                         f"1B/1.5B models) or set it to 0")
    last_wait = time.time()
    while pending or any(slots.values()):
        for g in gpus:
            slots[g] = [(n, p) for n, p in slots[g] if p.poll() is None or print(
                f"[{time.strftime('%H:%M:%S')}] finished {n} rc={p.returncode}", flush=True)]
        launched = False
        for g in sorted(gpus, key=lambda g: (len(slots[g]), gpus.index(g))):
            if not pending:
                break
            if len(slots[g]) >= args.max_parallel or time.time() - last[g] < settle or free_gb(g) < args.min_free_gb:
                continue
            j, out = pending.pop(0)
            os.makedirs(out, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
            cmd = [PY, args.script, *j["args"], "--out_dir", out]
            f = open(os.path.join(out, "stdout.log"), "w")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            slots[g].append((j["name"], p))
            last[g] = time.time()
            launched = True
            print(f"[{time.strftime('%H:%M:%S')}] start {j['name']} on gpu {g}", flush=True)
            time.sleep(5)   # spread model loading over a few seconds
        if pending and not launched and time.time() - last_wait > 600:
            print(f"[{time.strftime('%H:%M:%S')}] waiting: {len(pending)} pending, free GB "
                  + ", ".join(f"gpu {g} {free_gb(g):.0f}" for g in gpus)
                  + (f" (a job needs --min_free_gb {args.min_free_gb:g})" if args.min_free_gb > 0 else ""), flush=True)
            last_wait = time.time()
        if launched:
            last_wait = time.time()
        time.sleep(5 if launched else 15)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
