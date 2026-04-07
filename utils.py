"""
utils.py - Utilities: parallel runner, CPU info, helpers
"""
import subprocess
import time
import sys
import os
import platform
import shutil


def check_mpi_available() -> bool:
    """Check if mpirun is on PATH."""
    return shutil.which("mpirun") is not None or shutil.which("mpiexec") is not None


def get_mpi_command() -> str:
    if shutil.which("mpirun"):
        return "mpirun"
    if shutil.which("mpiexec"):
        return "mpiexec"
    return None


def run_parallel(processes: int, n: int, dataset_type: str = "random") -> tuple[float, str]:
    """
    Spawn parallel.py via mpirun and parse elapsed time.
    Returns (elapsed_time, error_message).
    """
    mpi_cmd = get_mpi_command()
    if not mpi_cmd:
        return None, "mpirun/mpiexec not found. Install OpenMPI or MPICH."

    # Locate parallel.py relative to this utils.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parallel_script = os.path.join(script_dir, "parallel.py")

    cmd = [
        mpi_cmd,
        "--oversubscribe",
        "-np", str(processes),
        sys.executable,
        parallel_script,
        "--size", str(n),
        "--dtype", dataset_type,
    ]

    try:
        wall_start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        wall_elapsed = time.perf_counter() - wall_start

        if result.returncode != 0:
            err = result.stderr.strip() or "Unknown MPI error"
            return None, f"MPI error (code {result.returncode}): {err[:300]}"

        # Parse MPI-reported time from stdout
        mpi_time = None
        for line in result.stdout.splitlines():
            if line.startswith("PARALLEL_TIME:"):
                mpi_time = float(line.split(":")[1])
                break

        # Fall back to wall time if MPI time not parsed
        elapsed = mpi_time if mpi_time is not None else wall_elapsed
        return elapsed, None

    except subprocess.TimeoutExpired:
        return None, "MPI process timed out (>120s)."
    except FileNotFoundError:
        return None, f"Could not launch '{mpi_cmd}'. Is MPI installed?"
    except Exception as e:
        return None, str(e)


def simulate_parallel(seq_time: float, processes: int, n: int) -> float:
    """
    Simulate parallel time using Amdahl's Law when MPI is unavailable.
    serial_fraction increases for small n.
    """
    base_serial = 0.05 + max(0, (5000 - n) / 100000)
    serial_fraction = min(base_serial, 0.9)
    parallel_fraction = 1 - serial_fraction
    # Amdahl's Law + small overhead term
    overhead = 0.001 * processes
    par_time = seq_time * (serial_fraction + parallel_fraction / processes) + overhead
    return par_time


def get_system_info() -> dict:
    info = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "mpi_available": check_mpi_available(),
        "mpi_command": get_mpi_command(),
    }
    try:
        import psutil
        info["cpu_freq_mhz"] = round(psutil.cpu_freq().current) if psutil.cpu_freq() else "N/A"
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu_percent"] = psutil.cpu_percent(interval=0.2)
    except ImportError:
        pass
    return info


def get_cpu_usage() -> float:
    """Return current CPU usage percent (requires psutil)."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.5)
    except ImportError:
        return -1.0
