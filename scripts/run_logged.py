"""Record an experiment command, output, process RSS and Linux cgroup memory."""
import argparse
import datetime as dt
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('--log-dir', type=Path, required=True)
p.add_argument('--interval', type=float, default=5)
p.add_argument('--memory-gib', type=float, help='child address-space limit (Linux RLIMIT_AS), not physical RSS')
p.add_argument('--cgroup-limit-gib', type=float, help='terminate child if sampled cgroup memory reaches this level')
p.add_argument('--ppu-smi', type=Path, help='sample PPU memory and utilization with compatible nvidia-smi')
p.add_argument('--nice', type=int, default=0, help='child niceness increment, 0-19')
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
if not cmd or a.interval <= 0 or not 0 <= a.nice <= 19 or (a.memory_gib is not None and a.memory_gib <= 0) or (a.cgroup_limit_gib is not None and a.cgroup_limit_gib <= 0):
    p.error('provide a command, positive interval/limit and niceness in 0..19')
if a.memory_gib is not None and sys.platform != 'linux':
    p.error('--memory-gib is supported only on Linux')
a.log_dir.mkdir(parents=True, exist_ok=False)
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
metadata = {'command':cmd, 'cwd':os.getcwd(), 'started_utc':now(), 'interval_seconds':a.interval, 'address_space_limit_gib':a.memory_gib, 'nice_increment':a.nice, 'cgroup_limit_gib':a.cgroup_limit_gib, 'ppu_smi':str(a.ppu_smi) if a.ppu_smi else None}
(a.log_dir/'run.json').write_text(json.dumps(metadata, indent=2))
start = time.monotonic()
with (a.log_dir/'console.log').open('w', buffering=1) as log, (a.log_dir/'resources.csv').open('w', buffering=1) as samples:
    def limit_child():
        if a.memory_gib is not None:
            limit = int(a.memory_gib * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        if a.nice:
            os.nice(a.nice)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                preexec_fn=limit_child if a.memory_gib is not None or a.nice else None)
    except Exception as exc:
        metadata.update(ended_utc=now(), elapsed_seconds=time.monotonic()-start, launch_error=str(exc))
        (a.log_dir/'run.json').write_text(json.dumps(metadata, indent=2))
        raise
    samples.write('elapsed_seconds,root_rss_kib,cgroup_current_bytes\n')
    stop = threading.Event()
    ppu_file = (a.log_dir/'ppu.csv').open('w', buffering=1) if a.ppu_smi else None
    if ppu_file: ppu_file.write('elapsed_seconds,memory_used_mib,memory_total_mib,utilization_percent\n')
    def monitor():
        last_ppu = -10.0
        while not stop.is_set():
            rss = ''
            try:
                for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'): rss = line.split()[1]
            except OSError: pass
            try: current = Path('/sys/fs/cgroup/memory.current').read_text().strip()
            except OSError: current = ''
            elapsed = time.monotonic()-start
            samples.write(f'{elapsed:.2f},{rss},{current}\n')
            if ppu_file and elapsed-last_ppu >= 5:
                last_ppu = elapsed
                try:
                    q = subprocess.run([str(a.ppu_smi), '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
                    fields = [x.strip() for x in q.stdout.strip().splitlines()[0].split(',')]
                    if len(fields) == 3 and all(x.isdigit() for x in fields):
                        ppu_file.write(f'{elapsed:.2f},{fields[0]},{fields[1]},{fields[2]}\n')
                except (OSError, subprocess.SubprocessError, IndexError):
                    pass
            if a.cgroup_limit_gib is not None and current and int(current) >= a.cgroup_limit_gib * 1024**3:
                metadata['terminated_for_cgroup_memory_bytes'] = int(current)
                print(f'cgroup memory guard: terminating child at {current} bytes', flush=True)
                proc.terminate()
                break
            stop.wait(a.interval)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for line in proc.stdout:
            log.write(line)
            print(line, end='', flush=True)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        code = proc.wait()
        raise
    finally:
        stop.set(); thread.join()
        if ppu_file: ppu_file.close()
        metadata.update(ended_utc=now(), elapsed_seconds=time.monotonic()-start, exit_code=proc.returncode)
        (a.log_dir/'run.json').write_text(json.dumps(metadata, indent=2))
sys.exit(code)
