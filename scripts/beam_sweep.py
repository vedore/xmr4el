import subprocess
import sys
from datetime import datetime

def log(msg, color="cyan"):
    colors = {
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color,'')}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}{colors['reset']}")

def run_tests(model_path, test_path, beam_start=5, beam_end=25, topk=10):
    for beam in range(beam_start, beam_end + 1, 5):
        log(f"Running test with beam_size={beam}", "yellow")
        cmd = [
            "python3",
            "scripts/evaluate.py",
            "-xmodel_path", model_path,
            "-test_path", test_path,
            "-beam_size", str(beam),
            "-topk", str(topk)
        ]
        try:
            subprocess.run(cmd, check=True)
            log(f"Completed beam_size={beam}", "green")
        except subprocess.CalledProcessError:
            log(f"Test failed for beam_size={beam}", "red")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python scripts/beam_sweep.py <model_path> <test_path> <beam_start> <beam_end> <topk>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_path = sys.argv[2]
    beam_start = int(sys.argv[3])
    beam_end = int(sys.argv[4])
    topk = int(sys.argv[5])

    run_tests(model_path, test_path, beam_start, beam_end, topk)