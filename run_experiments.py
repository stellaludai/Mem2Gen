#!/usr/bin/env python3
"""
Multi-GPU experiment scheduler.
Distributes training jobs across available GPUs.
Supports both single-fact and multi-fact training.

Usage:
    # Auto-detect training type from config path
    python run_experiments.py --manifest configs/experiment_configs/manifest.yaml --gpus 0,1,2,3
    python run_experiments.py --config_dir configs/multi_experiment_configs --gpus 0,1,2,3
    
    # Explicitly specify training script
    python run_experiments.py --config_dir configs/experiment_configs --gpus 0,1 --script train_single_fact.py
    python run_experiments.py --config_dir configs/multi_experiment_configs --gpus 0,1 --script train_multi_fact.py
    
    # Filter by name pattern
    python run_experiments.py --config_dir configs/experiment_configs --filter "amzn_qwen" --gpus 0,1
"""

import os
import sys
import yaml
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from queue import Queue
from threading import Thread, Lock
import signal

@dataclass
class TrainingJob:
    """Represents a single training job."""
    name: str
    config_path: str
    training_script: str = "train_single_fact.py"  # Default to single-fact
    gpu_id: int = -1
    status: str = "pending"  # pending, running, completed, failed
    process: Optional[subprocess.Popen] = None
    output_dir: Optional[str] = None


class GPUScheduler:
    """Scheduler to distribute jobs across GPUs."""
    
    def __init__(self, gpu_ids: List[int], log_dir: str = "./experiment_logs", default_script: Optional[str] = None):
        self.gpu_ids = gpu_ids
        self.available_gpus = Queue()
        for gpu_id in gpu_ids:
            self.available_gpus.put(gpu_id)
        
        self.jobs: List[TrainingJob] = []
        self.running_jobs: Dict[int, TrainingJob] = {}  # gpu_id -> job
        self.lock = Lock()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stop_flag = False
        self.default_script = default_script  # User-specified default script
        
    def _detect_training_script(self, config_path: str) -> str:
        """
        Auto-detect which training script to use based on config path or content.
        
        Priority:
        1. If user specified a default script, use it
        2. If config path contains 'multi_experiment' or 'multi_fact', use train_multi_fact.py
        3. Try to read config and check for 'task_type' field (multi-fact specific)
        4. Default to train_single_fact.py
        """
        if self.default_script:
            return self.default_script
        
        # Check path for hints
        config_path_lower = config_path.lower()
        if 'multi_experiment' in config_path_lower or 'multi_fact' in config_path_lower:
            return "train_multi_fact.py"
        
        # Try reading config to check for task_type field
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Multi-fact configs have dataset.task_type field
                if config.get('dataset', {}).get('task_type') in ['chaining_tasks', 'counting_tasks', 'intersection_tasks']:
                    return "train_multi_fact.py"
        except Exception:
            pass  # If we can't read the config, fall back to default
        
        # Default to single-fact
        return "train_single_fact.py"
    
    def add_job(self, name: str, config_path: str, training_script: Optional[str] = None):
        """Add a job to the queue."""
        if training_script is None:
            training_script = self._detect_training_script(config_path)
        
        job = TrainingJob(name=name, config_path=config_path, training_script=training_script)
        self.jobs.append(job)
        
    def load_jobs_from_manifest(self, manifest_path: str, filter_pattern: Optional[str] = None):
        """Load jobs from manifest YAML file."""
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        for config_info in manifest.get("configs", []):
            if filter_pattern and filter_pattern not in config_info["name"]:
                continue
            self.add_job(config_info["name"], config_info["path"])
            
    def load_jobs_from_directory(self, config_dir: str, filter_pattern: Optional[str] = None):
        """Load all YAML configs from a directory."""
        config_path = Path(config_dir)
        for yaml_file in sorted(config_path.glob("*.yaml")):
            if yaml_file.name == "manifest.yaml":
                continue
            name = yaml_file.stem
            if filter_pattern and filter_pattern not in name:
                continue
            self.add_job(name, str(yaml_file))
    
    def _run_job(self, job: TrainingJob, gpu_id: int):
        """Run a single training job."""
        job.gpu_id = gpu_id
        job.status = "running"
        
        # Create log file
        log_file = self.log_dir / f"{job.name}.log"
        
        # Build command
        cmd = [
            sys.executable,
            job.training_script,  # Use detected or specified script
            "--config", job.config_path,
            "--gpu", str(gpu_id)
        ]
        
        print(f"[GPU {gpu_id}] Starting: {job.name} (using {job.training_script})")
        print(f"[GPU {gpu_id}] Log file: {log_file}")
        
        try:
            with open(log_file, 'w') as f:
                job.process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                )
            
            with self.lock:
                self.running_jobs[gpu_id] = job
                
            # Wait for completion
            job.process.wait()
            
            if job.process.returncode == 0:
                job.status = "completed"
                print(f"[GPU {gpu_id}] Completed: {job.name}")
            else:
                job.status = "failed"
                print(f"[GPU {gpu_id}] Failed: {job.name} (exit code: {job.process.returncode})")
                
        except Exception as e:
            job.status = "failed"
            print(f"[GPU {gpu_id}] Error running {job.name}: {e}")
        
        finally:
            with self.lock:
                if gpu_id in self.running_jobs:
                    del self.running_jobs[gpu_id]
            self.available_gpus.put(gpu_id)
    
    def run_all(self):
        """Run all jobs, distributing across GPUs."""
        print(f"Total jobs: {len(self.jobs)}")
        print(f"Available GPUs: {self.gpu_ids}")
        
        # Print job breakdown by script type
        single_jobs = [j for j in self.jobs if "single" in j.training_script]
        multi_jobs = [j for j in self.jobs if "multi" in j.training_script]
        print(f"Single-fact jobs: {len(single_jobs)}")
        print(f"Multi-fact jobs: {len(multi_jobs)}")
        print("-" * 50)
        
        threads = []
        job_index = 0
        
        while job_index < len(self.jobs) or threads:
            # Clean up completed threads
            threads = [t for t in threads if t.is_alive()]
            
            # Check stop flag
            if self.stop_flag:
                print("Stop flag set, waiting for running jobs to complete...")
                for t in threads:
                    t.join()
                break
            
            # Start new jobs if GPUs available
            while job_index < len(self.jobs) and not self.available_gpus.empty():
                gpu_id = self.available_gpus.get()
                job = self.jobs[job_index]
                job_index += 1
                
                t = Thread(target=self._run_job, args=(job, gpu_id))
                t.start()
                threads.append(t)

                print(f"→ Submitted task to GPU {gpu_id} ({len(self.jobs) - job_index} remaining)")
            
            time.sleep(1)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print job execution summary."""
        print("\n" + "=" * 50)
        print("EXECUTION SUMMARY")
        print("=" * 50)
        
        completed = [j for j in self.jobs if j.status == "completed"]
        failed = [j for j in self.jobs if j.status == "failed"]
        pending = [j for j in self.jobs if j.status == "pending"]
        
        # Breakdown by training type
        single_completed = [j for j in completed if "single" in j.training_script]
        multi_completed = [j for j in completed if "multi" in j.training_script]
        single_failed = [j for j in failed if "single" in j.training_script]
        multi_failed = [j for j in failed if "multi" in j.training_script]
        
        print(f"Total: {len(self.jobs)}")
        print(f"  Completed: {len(completed)} (Single: {len(single_completed)}, Multi: {len(multi_completed)})")
        print(f"  Failed: {len(failed)} (Single: {len(single_failed)}, Multi: {len(multi_failed)})")
        print(f"  Pending: {len(pending)}")
        
        if failed:
            print("\nFailed jobs:")
            for job in failed:
                print(f"  - {job.name} ({job.training_script})")
        
        # Save summary to file
        summary = {
            "total": len(self.jobs),
            "completed": [{"name": j.name, "script": j.training_script} for j in completed],
            "failed": [{"name": j.name, "script": j.training_script} for j in failed],
            "pending": [{"name": j.name, "script": j.training_script} for j in pending],
            "breakdown": {
                "single_fact": {
                    "completed": len(single_completed),
                    "failed": len(single_failed),
                },
                "multi_fact": {
                    "completed": len(multi_completed),
                    "failed": len(multi_failed),
                }
            }
        }
        summary_path = self.log_dir / "summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        print(f"\nSummary saved to: {summary_path}")
    
    def stop(self):
        """Signal to stop scheduling new jobs."""
        self.stop_flag = True
        print("\nStopping scheduler (will wait for running jobs)...")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU experiment scheduler for single-fact and multi-fact training")
    parser.add_argument("--manifest", type=str, help="Path to manifest YAML file")
    parser.add_argument("--config_dir", type=str, help="Directory containing config YAML files")
    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated GPU IDs (e.g., 0,1,2,3)")
    parser.add_argument("--filter", type=str, default=None, help="Filter pattern for experiment names")
    parser.add_argument("--log_dir", type=str, default="./experiment_logs", help="Directory for log files")
    parser.add_argument("--script", type=str, default=None, 
                        choices=["train_single_fact.py", "train_multi_fact.py"],
                        help="Explicitly specify training script (auto-detect if not provided)")
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"Using GPUs: {gpu_ids}")
    
    if args.script:
        print(f"Training script: {args.script} (explicitly specified)")
    else:
        print("Training script: auto-detect from config")
    
    # Create scheduler
    scheduler = GPUScheduler(gpu_ids, log_dir=args.log_dir, default_script=args.script)
    
    # Load jobs
    if args.manifest:
        scheduler.load_jobs_from_manifest(args.manifest, args.filter)
    elif args.config_dir:
        scheduler.load_jobs_from_directory(args.config_dir, args.filter)
    else:
        print("Error: Either --manifest or --config_dir must be specified")
        sys.exit(1)
    
    if not scheduler.jobs:
        print("No jobs to run!")
        sys.exit(0)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        scheduler.stop()
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run all jobs
    scheduler.run_all()


if __name__ == "__main__":
    main()