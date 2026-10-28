"""Per-stage resource summaries -> resources.json (docs/design/tui.md §8). A daemon thread samples every
`interval` seconds; stage start and end sample too, so a stage shorter than the interval still has two samples.
Raw samples are not kept.

Scope and units (null = not measurable here):
- cpu_avg_percent: CPU time of this process (all threads) and of its exited, waited-for children over the stage's
  wall time; 100 = one core busy. Exact (os.times), not sampled.
- rss_peak_bytes / rss_avg_bytes: resident memory of this process, sampled (Linux /proc, macOS proc_pidinfo).
  The peak is exact when the stage raised the process's lifetime peak (getrusage), else the sampled maximum.
- gpu_util_avg_percent: CUDA only, whole device (other processes included), sampled via NVML (the optional
  `nvml` extra, nvidia-ml-py); null without it or without an NVML driver library. Queried by device index, so it
  never initialises CUDA.
- torch_gpu_allocated_peak_bytes: bytes held by PyTorch tensors of this process on its GPU, the caching
  allocator's "allocated" count. CUDA: torch.cuda.max_memory_allocated, exact, reset at stage start (only when
  CUDA is initialised: never creates a context). MPS: torch.mps.current_allocated_memory, sampled maximum. Not the
  process's total GPU memory (excludes the allocator's reserved-but-free cache, the CUDA context, other libraries)
  and not device memory as nvidia-smi shows it. Null on CPU or while CUDA is not initialised.
"""
import ctypes
import json
import os
import platform
import resource
import threading

import torch

SCHEMA_VERSION = 1
LINUX, MAC = platform.system() == "Linux", platform.system() == "Darwin"
MAXRSS_UNIT = 1 if MAC else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
PAGE = os.sysconf("SC_PAGE_SIZE") if LINUX else 0
PROC_PIDTASKINFO = 4  # macOS: struct proc_taskinfo, 96 bytes, pti_resident_size is its second uint64
_libc = ctypes.CDLL(None) if MAC else None
STATUS = {"complete": "completed", "failed": "failed", "stopped": "stopped"}
SCOPE = {"cpu": "this process (all threads) + exited children; 100 = one core", "rss": "this process"}
GPU_SCOPE = {"cuda": {"gpu_util": "whole CUDA device, sampled (NVML)",
                      "gpu_memory": "this process, PyTorch tensors: torch.cuda.max_memory_allocated, exact"},
             "mps": {"gpu_util": None,
                     "gpu_memory": "this process, PyTorch tensors: torch.mps.current_allocated_memory, sampled max"}}


def rss_now():
    """Resident bytes of this process now, None when unreadable."""
    try:
        if LINUX:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * PAGE
        if MAC:
            info = (ctypes.c_uint64 * 12)()
            if _libc.proc_pidinfo(os.getpid(), PROC_PIDTASKINFO, ctypes.c_uint64(0), info,
                                  ctypes.sizeof(info)) == ctypes.sizeof(info):
                return info[1]
    except (OSError, ValueError, AttributeError):
        pass
    return None


def cpu_seconds():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def maxrss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * MAXRSS_UNIT


class ResourceMonitor:
    """Aggregates samples by top-level stage (one at a time: stages are sequential); `finish` appends the
    stage's summary to resources.json (write-then-rename). `close` stops the sampling thread."""

    def __init__(self, path, device, interval):
        self.path, self.device, self.interval = path, torch.device(device), interval
        self.lock, self.done = threading.Lock(), threading.Event()
        self.current, self.stages, self.nvml = None, [], self.device.type == "cuda"
        threading.Thread(target=self.loop, daemon=True, name="resources").start()

    def loop(self):
        while not self.done.wait(self.interval):
            self.sample()

    def gpu_util(self):
        if not self.nvml:
            return None
        try:
            # an int index: a device without one makes torch call current_device(), which initialises CUDA
            return float(torch.cuda.utilization(self.device.index or 0))
        except Exception:  # no nvidia-ml-py / NVML: never asked again
            self.nvml = False
            return None

    def gpu_mem_sample(self):
        return torch.mps.current_allocated_memory() if self.device.type == "mps" else None

    def sample(self):
        """One sample of the stage open when it starts; dropped if that stage ended while measuring (the lock is
        not held across the measurement: an NVML query may be slow)."""
        with self.lock:
            s = self.current
        if s is None:
            return
        rss, util, gpu_mem = rss_now(), self.gpu_util(), self.gpu_mem_sample()
        with self.lock:
            if self.current is not s:
                return
            s["n"] += 1
            if rss is not None:
                s["rss"].append(rss)
            if util is not None:
                s["util"].append(util)
            if gpu_mem is not None:
                s["gpu_mem"] = max(s["gpu_mem"] or 0, gpu_mem)

    def start(self, key):
        cuda = self.device.type == "cuda" and torch.cuda.is_initialized()  # never creates a CUDA context
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        with self.lock:
            self.current = {"key": key, "cpu": cpu_seconds(), "maxrss": maxrss(), "n": 0, "rss": [], "util": [],
                            "gpu_mem": None}
        self.sample()

    def finish(self, key, phase, seconds):
        self.sample()
        with self.lock:
            s, self.current = self.current, None
        if s is None or s["key"] != key:
            return
        stage, split, _ = key
        peak = max(s["rss"], default=None)
        if (end := maxrss()) > s["maxrss"]:
            peak = end  # the lifetime peak rose during this stage: that is the stage's exact peak
        gpu_mem = s["gpu_mem"]
        if self.device.type == "cuda" and torch.cuda.is_initialized():
            gpu_mem = torch.cuda.max_memory_allocated(self.device)
        avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
        self.stages.append({
            "stage": stage, **({"split": split} if split else {}), "status": STATUS[phase],
            "duration_seconds": seconds, "samples": s["n"],
            "resources": {"cpu_avg_percent": round(100 * (cpu_seconds() - s["cpu"]) / seconds, 1) if seconds > 0
                          else None, "rss_peak_bytes": peak, "rss_avg_bytes": int(sum(s["rss"]) / len(s["rss"])) if s["rss"] else None,
                          "gpu_util_avg_percent": avg(s["util"]), "torch_gpu_allocated_peak_bytes": gpu_mem}})
        doc = {"v": SCHEMA_VERSION, "device": str(self.device), "interval_seconds": self.interval, "scope": {**SCOPE, **GPU_SCOPE.get(self.device.type, {"gpu_util": None, "gpu_memory": None})},
               "stages": self.stages}
        with open(self.path + ".tmp", "w") as f:
            json.dump(doc, f, indent=2)
        os.replace(self.path + ".tmp", self.path)

    def close(self):
        self.done.set()
