#!/usr/bin/env python3
"""Monitor idle NVIDIA GPUs and continuously run CUDA work on claimed devices."""
import ctypes as c
import csv
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / '.local' / 'share' / 'gpu-idle-monitor'
ROOT.mkdir(parents=True, exist_ok=True)
log = logging.getLogger('gpu-idle-monitor')
log.setLevel(logging.INFO)
h = RotatingFileHandler(ROOT / 'monitor.log', maxBytes=2*1024*1024, backupCount=3)
h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
log.addHandler(h)
log.addHandler(logging.StreamHandler())


def worker(uuid):
    # A kernel-enforced lock prevents duplicate workers, including across monitor crashes.
    lock = open(ROOT / ('worker-' + uuid + '.lock'), 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error('worker already running gpu=%s', uuid)
        return 2
    rt = c.CDLL('libcudart.so.11.0')
    blas = c.CDLL('libcublas.so.11')
    rt.cudaSetDevice.argtypes = [c.c_int]
    rt.cudaMalloc.argtypes = [c.POINTER(c.c_void_p), c.c_size_t]
    rt.cudaMemset.argtypes = [c.c_void_p, c.c_int, c.c_size_t]
    rt.cudaDeviceSynchronize.argtypes = []
    blas.cublasCreate_v2.argtypes = [c.POINTER(c.c_void_p)]
    blas.cublasSgemm_v2.argtypes = [c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int, c.POINTER(c.c_float), c.c_void_p, c.c_int, c.c_void_p, c.c_int, c.POINTER(c.c_float), c.c_void_p, c.c_int]
    def check(code, name):
        if code != 0: raise RuntimeError('%s returned %d' % (name, code))
    check(rt.cudaSetDevice(0), 'cudaSetDevice')
    handle = c.c_void_p()
    check(blas.cublasCreate_v2(c.byref(handle)), 'cublasCreate')
    n = 2048
    size = n*n*4
    arrays = [c.c_void_p() for _ in range(3)]
    for a in arrays:
        check(rt.cudaMalloc(c.byref(a), size), 'cudaMalloc')
        check(rt.cudaMemset(a, 1, size), 'cudaMemset')
    # Each chunk remains resident; never count other processes toward our target.
    reserves = []
    next_increase = 0.0
    alpha, beta = c.c_float(1), c.c_float(0)
    while True:
        if time.monotonic() >= next_increase:
            next_increase = time.monotonic() + 60
            try:
                row = next(r for r in query('uuid,memory.total,memory.free', 'gpu') if r[0].strip() == uuid)
                total, free = (int(v.strip()) for v in row[1:])
                own_rows = [r for r in query('gpu_uuid,pid,used_memory', 'compute-apps')
                            if r[0].strip() == uuid and int(r[1].strip()) == os.getpid()]
                if len(own_rows) != 1:
                    raise ValueError('own GPU memory unavailable or ambiguous')
                own = int(own_rows[0][2].strip())
                target = total * 80 // 100
                initial = own
                failures = 0
                while True:
                    amount = allocation_mib(target, own, free)
                    if not amount:
                        break
                    reserve = c.c_void_p()
                    code = rt.cudaMalloc(c.byref(reserve), amount * 1024 * 1024)
                    if code:
                        failures += 1
                        log.warning('memory allocation failed gpu=%s target=%sMiB own_estimate=%sMiB free_estimate=%sMiB chunk=%sMiB cudaMalloc=%s', uuid, target, own, free, amount, code)
                        break
                    reserves.append(reserve)
                    own += amount
                    free -= amount
                log.info('memory retry gpu=%s target=%sMiB own_before=%sMiB own_estimate=%sMiB free_estimate=%sMiB chunks=%s failures=%s margin=1024MiB', uuid, target, initial, own, free, len(reserves), failures)
            except Exception:
                log.exception('memory retry failed gpu=%s; retaining allocations and continuing compute', uuid)
        check(blas.cublasSgemm_v2(handle, 0, 0, n, n, n, c.byref(alpha), arrays[0], n, arrays[1], n, c.byref(beta), arrays[2], n), 'cublasSgemm')
        check(rt.cudaDeviceSynchronize(), 'cudaDeviceSynchronize')


def allocation_mib(target, own, free):
    return max(0, min(256, target - own, free - 1024))


def query(fields, kind):
    p = subprocess.run(['nvidia-smi', '--query-'+kind+'='+fields, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    return list(csv.reader(p.stdout.splitlines(), skipinitialspace=True))


def snapshot():
    rows = query('index,uuid,utilization.gpu,memory.used', 'gpu')
    if not rows: raise ValueError('no GPUs returned')
    gpus = {}
    for row in rows:
        if len(row) != 4: raise ValueError('invalid GPU row')
        index, uuid, util, mem = [v.strip() for v in row]
        if not uuid.startswith('GPU-') or uuid in gpus: raise ValueError('invalid GPU UUID')
        index, util, mem = int(index), int(util), int(mem)
        if index < 0 or not 0 <= util <= 100 or mem < 0: raise ValueError('invalid GPU values')
        gpus[uuid] = {'index': index, 'util': util, 'mem': mem, 'pids': []}
    for row in query('gpu_uuid,pid', 'compute-apps'):
        if len(row) != 2: raise ValueError('invalid process row')
        uuid, pid = [v.strip() for v in row]
        if uuid not in gpus: raise ValueError('unknown process GPU UUID')
        gpus[uuid]['pids'].append(int(pid))
    return gpus


def idle(g):
    return not g['pids'] and g['util'] == 0 and g['mem'] <= 256


def save_claims(claims):
    tmp = ROOT / 'claims.tmp'
    with open(tmp, 'w') as f:
        json.dump(sorted(claims), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ROOT / 'claims.json')
    fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main():
    lock = open(ROOT / 'monitor.lock', 'a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error('another monitor holds lock; exiting')
        return 1
    state = ROOT / 'claims.json'
    if state.exists():
        with open(state) as f: saved = json.load(f)
        if not isinstance(saved, list) or any(not isinstance(x, str) or not x.startswith('GPU-') or '/' in x for x in saved):
            raise ValueError('invalid claim state')
        claims = set(saved)
    else: claims = set()
    exclusion_file = ROOT / 'excluded.json'
    if exclusion_file.exists():
        with open(exclusion_file) as f: excluded = json.load(f)
        if not isinstance(excluded, list) or any(not isinstance(x, str) or not x.startswith('GPU-') or '/' in x for x in excluded):
            raise ValueError('invalid exclusion state')
        excluded = set(excluded)
    else:
        excluded = set()
    claims -= excluded
    log.info('persistent exclusions=%s', sorted(excluded))
    children, next_retry = {}, {}
    stopping = False
    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    def launch(uuid):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = uuid
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', uuid], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children[uuid] = p
        next_retry[uuid] = time.monotonic() + 15
        log.info('worker launched gpu=%s pid=%s', uuid, p.pid)
    log.info('monitor started pid=%s restored_claims=%s', os.getpid(), sorted(claims))
    next_sample = 0
    try:
        while not stopping:
            now = time.monotonic()
            for uuid, p in list(children.items()):
                if p.poll() is not None:
                    log.error('worker exited gpu=%s pid=%s exit=%s; will retry', uuid, p.pid, p.returncode)
                    del children[uuid]
            for uuid in sorted(claims):
                if stopping: break
                if uuid not in children and now >= next_retry.get(uuid, 0):
                    try: launch(uuid)
                    except Exception:
                        next_retry[uuid] = time.monotonic() + 15
                        log.exception('worker launch failed gpu=%s; will retry', uuid)
            if now >= next_sample:
                next_sample = now + 60
                try:
                    gpus = snapshot()
                    for uuid, g in gpus.items():
                        own = children.get(uuid)
                        foreign = [pid for pid in g['pids'] if own is None or pid != own.pid]
                        log.info('status gpu=%s index=%s utilization=%s%% memory=%sMiB compute_pids=%s own_pid=%s claimed=%s', uuid, g['index'], g['util'], g['mem'], g['pids'], own.pid if own else None, uuid in claims)
                        if uuid in claims and foreign:
                            log.warning('external compute detected gpu=%s pids=%s; continuing', uuid, foreign)
                    for uuid in claims - gpus.keys():
                        log.warning('claimed GPU missing gpu=%s; continuing recovery', uuid)
                    for uuid, g in gpus.items():
                        if stopping: break
                        if uuid in excluded or uuid in claims or not idle(g): continue
                        latest = snapshot()
                        if uuid not in latest or not idle(latest[uuid]):
                            log.info('launch skipped gpu=%s reason=recheck not idle', uuid)
                            continue
                        # Persist before starting, so a crash cannot lose an established claim.
                        save_claims(claims | {uuid})
                        claims.add(uuid)
                        log.info('gpu claimed gpu=%s index=%s', uuid, latest[uuid]['index'])
                except Exception:
                    log.exception('GPU check or launch failed; retaining existing workers and skipping new claims')
            time.sleep(1)
    finally:
        # Preserve claims across service restarts; an explicit reset removes the state file.
        for uuid, p in children.items():
            if p.poll() is None: p.terminate()
        for uuid, p in children.items():
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            log.info('worker stopped gpu=%s pid=%s exit=%s', uuid, p.pid, p.returncode)
        log.info('monitor stopped pid=%s', os.getpid())
    return 0

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        try: sys.exit(worker(sys.argv[2]))
        except Exception:
            log.exception('worker failure pid=%s', os.getpid())
            sys.exit(1)
    else: sys.exit(main())
