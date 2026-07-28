"""Process-monitoring runner.

Spawns a command as a black-box subprocess, samples its process tree and
system-wide metrics on an interval, and writes a CSV plus a markdown report.
Standalone: imports only the standard library and psutil.

Usage:
    python -m runner.monitor [--interval SEC] [--duration SEC] [--out DIR] -- <command...>
"""

import argparse
import csv
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

FIELDS = [
    "elapsed",
    "rss_bytes",
    "cpu_percent",
    "threads",
    "fds",
    "inet_sockets",
    "sys_cpu_percent",
    "sys_mem_percent",
    "net_bytes_sent",
    "net_bytes_recv",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m runner.monitor",
        description="Run a command and sample its process tree until it exits.",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="sampling interval in seconds (default: 1.0)")
    parser.add_argument("--duration", type=float, default=None, help="terminate the command after this many seconds")
    parser.add_argument("--out", default="reports", help="reports directory (default: reports)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be > 0")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("no command given (put it after --)")
    return args


def inet_sockets(proc):
    # Process.net_connections() replaced connections() in psutil 6.0.
    if hasattr(proc, "net_connections"):
        return proc.net_connections(kind="inet")
    return proc.connections(kind="inet")


def collect_sample(root, cache, elapsed):
    """Sample the process tree; returns None if the whole tree is gone.

    ``cache`` maps pid -> psutil.Process and persists across samples, so each
    cpu_percent() measures since the previous sample instead of resetting to a
    fresh 0.0 baseline on a brand-new Process object every time.
    """
    procs = {root.pid: root}
    try:
        for proc in root.children(recursive=True):
            cached = cache.get(proc.pid)
            # psutil equality includes creation time, guarding against pid reuse.
            procs[proc.pid] = cached if cached == proc else proc
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    cache.clear()
    cache.update(procs)
    rss = cpu = threads = fds = sockets = seen = 0
    for proc in procs.values():
        # Read into locals first, commit only on full success, so a process
        # failing partway through doesn't leave a half-counted sample.
        try:
            with proc.oneshot():
                p_rss = proc.memory_info().rss
                # First cpu_percent() call per process is a 0.0 baseline.
                p_cpu = proc.cpu_percent(interval=None)
                p_threads = proc.num_threads()
                p_fds = proc.num_fds()
                p_sockets = len(inet_sockets(proc))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue  # exited mid-sample: normal lifecycle, skip this process
        rss += p_rss
        cpu += p_cpu
        threads += p_threads
        fds += p_fds
        sockets += p_sockets
        seen += 1
    if seen == 0:
        return None
    net = psutil.net_io_counters()
    return {
        "elapsed": round(elapsed, 2),
        "rss_bytes": rss,
        "cpu_percent": round(cpu, 1),
        "threads": threads,
        "fds": fds,
        "inet_sockets": sockets,
        "sys_cpu_percent": psutil.cpu_percent(interval=None),
        "sys_mem_percent": psutil.virtual_memory().percent,
        "net_bytes_sent": net.bytes_sent,
        "net_bytes_recv": net.bytes_recv,
    }


def stop_child(child, root):
    """Terminate the whole tree; the child may not forward SIGTERM to its own."""
    if child.poll() is not None:
        return
    try:
        descendants = root.children(recursive=True)
    except psutil.Error:
        descendants = []
    child.terminate()
    for proc in descendants:
        try:
            proc.terminate()
        except psutil.Error:
            pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    for proc in psutil.wait_procs(descendants, timeout=5)[1]:
        try:
            proc.kill()
        except psutil.Error:
            pass


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def write_csv(path, samples):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(samples)


def summary_table(samples):
    specs = [
        ("Process RSS (MB)", lambda s: s["rss_bytes"] / 2**20, "{:.1f}"),
        ("Process CPU %", lambda s: s["cpu_percent"], "{:.1f}"),
        ("Threads", lambda s: s["threads"], "{:.0f}"),
        ("Open FDs", lambda s: s["fds"], "{:.0f}"),
        ("Inet sockets", lambda s: s["inet_sockets"], "{:.0f}"),
        ("System CPU %", lambda s: s["sys_cpu_percent"], "{:.1f}"),
        ("System memory %", lambda s: s["sys_mem_percent"], "{:.1f}"),
    ]
    lines = ["| Metric | Min | Mean | Peak |", "| --- | --- | --- | --- |"]
    for label, get, fmt in specs:
        values = [get(s) for s in samples]
        cells = (fmt.format(v) for v in (min(values), statistics.fmean(values), max(values)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def timeline_chart(samples, height=8, width=60):
    """Block chart of process RSS over the run, one column per time bucket.
    Fenced as a code block so it renders monospaced in any markdown viewer."""
    points = [(s["elapsed"], s["rss_bytes"] / 2**20) for s in samples]
    if len(points) > width:
        last = len(points) - 1
        points = [points[round(i * last / (width - 1))] for i in range(width)]
    values = [v for _, v in points]
    lo, hi = min(values), max(values)
    span = hi - lo
    levels = [round((v - lo) / span * (height - 1)) if span else 0 for v in values]
    lines = ["```"]
    for row in range(height - 1, -1, -1):
        if row == height - 1:
            label = f"{hi:8.1f} |"
        elif row == 0:
            label = f"{lo:8.1f} |"
        else:
            label = " " * 9 + "|"
        lines.append(label + "".join("█" if level >= row else " " for level in levels))
    lines.append(" " * 9 + "+" + "-" * len(levels))
    end = f"{points[-1][0]:.0f}s"
    lines.append(" " * 10 + "0s" + end.rjust(max(len(levels) - 2, len(end))))
    lines.append("```")
    return lines


def write_report(path, command, started, ended, exit_code, interval, samples):
    lines = [
        "# Monitor report",
        "",
        f"- Command: `{shlex.join(command)}`",
        f"- Started: {started}",
        f"- Ended: {ended}",
        f"- Wall duration: {(datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds():.1f} s",
        f"- Child exit code: {exit_code}",
        f"- Samples: {len(samples)} at {interval} s interval",
        "",
    ]
    if samples:
        sent = samples[-1]["net_bytes_sent"] - samples[0]["net_bytes_sent"]
        recv = samples[-1]["net_bytes_recv"] - samples[0]["net_bytes_recv"]
        lines += ["## Summary", ""]
        lines += summary_table(samples)
        lines += [
            "",
            "## Network",
            "",
            f"- Bytes sent: {human_bytes(sent)}",
            f"- Bytes received: {human_bytes(recv)}",
            "- Caveat: network counters are host-wide, not scoped to the monitored process.",
            "",
            "## Timeline — process RSS (MB)",
            "",
        ]
        lines += timeline_chart(samples)
    else:
        lines += ["No samples collected (child exited before it could be sampled).", ""]
    lines += [
        "",
        "## Notes",
        "",
        "- The first sample's CPU figures (per-process and system-wide) are 0.0 baselines,",
        "  which skews the Summary Min/Mean for the CPU rows low.",
        "- Network byte counts are host-wide, not process-scoped.",
        "- FD and socket counts are point-in-time snapshots at each sample.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    args = parse_args(argv)
    # Microsecond timestamp avoids two runs in the same second sharing a
    # directory; exist_ok=False makes any residual collision an error, not a
    # silent clobber.
    run_dir = Path(args.out) / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run_dir.mkdir(parents=True, exist_ok=False)

    started = datetime.now().isoformat(timespec="seconds")
    child = subprocess.Popen(args.command)
    root = psutil.Process(child.pid)
    print(f"monitor: pid {child.pid}, sampling every {args.interval} s", file=sys.stderr)

    start = time.monotonic()
    samples = []
    cache = {}

    def take_sample():
        sample = collect_sample(root, cache, time.monotonic() - start)
        if sample:
            samples.append(sample)

    take_sample()  # immediate t=0 sample, so short-lived children still get one
    try:
        while child.poll() is None:
            remaining = None
            if args.duration is not None:
                remaining = args.duration - (time.monotonic() - start)
                if remaining <= 0:
                    take_sample()  # final sample before tearing the tree down
                    stop_child(child, root)
                    break
            try:
                child.wait(timeout=min(args.interval, remaining) if remaining else args.interval)
            except subprocess.TimeoutExpired:
                take_sample()
    except KeyboardInterrupt:
        stop_child(child, root)
    ended = datetime.now().isoformat(timespec="seconds")

    write_csv(run_dir / "samples.csv", samples)
    write_report(run_dir / "report.md", args.command, started, ended, child.returncode, args.interval, samples)
    print(f"monitor: report written to {run_dir}", file=sys.stderr)
    # Map signal deaths (negative returncode) to the shell convention 128+signum,
    # so scripts don't see e.g. -15 wrap around to exit status 241.
    return 128 - child.returncode if child.returncode < 0 else child.returncode


if __name__ == "__main__":
    sys.exit(main())
