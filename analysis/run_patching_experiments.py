#!/usr/bin/env python3
"""
Multi-GPU patching experiment scheduler.
Distributes patching experiments across available GPUs.

Usage:
    # Run experiments from a manifest file
    python run_patching_experiments.py --manifest experiments_manifest.yaml --gpus 0,1,2,3
    
    # Auto-discover checkpoints from base directory
    python run_patching_experiments.py \
        --base_dir /cache/multi_fact_checkpoints \
        --models qwen2.5-3b \
        --tasks chaining counting intersection \
        --datasizes n1 n2 n3 \
        --gpus 0,1,2,3
    
    # Filter by specific patterns
    python run_patching_experiments.py \
        --base_dir /cache/multi_fact_checkpoints \
        --models qwen2.5-3b \
        --tasks chaining \
        --datasizes n1 \
        --filter "id520" \
        --gpus 0,1
"""

import os
import sys
import yaml
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread, Lock
import signal
from glob import glob


@dataclass
class PatchingJob:
    """Represents a single patching experiment job."""
    name: str
    ckpt_dir: str
    task_name: str
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    metric: str = "mrr"
    patching_position_type: str = "entity"  # 新增字段
    gpu_id: int = -1
    status: str = "pending"  # pending, running, completed, failed, skipped
    process: Optional[subprocess.Popen] = None
    error_msg: Optional[str] = None


class GPUScheduler:
    """Scheduler to distribute patching jobs across GPUs."""
    
    def __init__(self, gpu_ids: List[int], log_dir: str = "./patching_logs", 
                 script_path: str = "multi_fact_experiment.py"):
        self.gpu_ids = gpu_ids
        self.available_gpus = Queue()
        for gpu_id in gpu_ids:
            self.available_gpus.put(gpu_id)
        
        self.jobs: List[PatchingJob] = []
        self.running_jobs: Dict[int, PatchingJob] = {}  # gpu_id -> job
        self.lock = Lock()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stop_flag = False
        self.script_path = script_path

    def add_job(self, name: str, ckpt_dir: str, task_name: str, 
                base_model: str = "Qwen/Qwen2.5-3B-Instruct", 
                patching_position_type: str = "entity", **kwargs):
        """Add a patching job to the queue."""
        # Check if results already exist (检查对应patching_position_type的结果文件)
        result_file = Path(ckpt_dir) / f"results_self_patch_{patching_position_type}.npy"
        if result_file.exists():
            print(f"Skipping {name} (position_type={patching_position_type}): results already exist")
            job = PatchingJob(
                name=name, ckpt_dir=ckpt_dir, task_name=task_name,
                base_model=base_model, status="skipped", 
                patching_position_type=patching_position_type, **kwargs
            )
        else:
            job = PatchingJob(
                name=name, ckpt_dir=ckpt_dir, task_name=task_name,
                base_model=base_model, patching_position_type=patching_position_type, **kwargs
            )
        self.jobs.append(job)
        

    def discover_checkpoints(self, base_dir: str, models: List[str], 
                        tasks: List[str], datasizes: List[str],
                        patching_position_types: List[str] = None,  # 新增参数
                        filter_pattern: Optional[str] = None):
        """
        Auto-discover checkpoint directories from base directory structure.
        
        Expected structure:
        base_dir/
            {task}/
                {model}/
                    {datasize}/
                        # Format 1: multi_{task}_{model}_id{id}_{timestamp}/
                        # Format 2: multi_{task}_{model}_{datasize}_{timestamp}/
        """
        if patching_position_types is None:
            patching_position_types = ["entity"]
        
        base_path = Path(base_dir)
        
        for task in tasks:
            for model in models:
                for datasize in datasizes:
                    search_path = base_path / task / model / datasize
                    if not search_path.exists():
                        print(f"Warning: Path not found: {search_path}")
                        continue
                    
                    # Find all checkpoint directories matching both formats
                    pattern1 = f"multi_{task}_{model}_id*"
                    pattern2 = f"multi_{task}_{model}_{datasize}_*"
                    
                    ckpt_dirs = set()
                    ckpt_dirs.update(search_path.glob(pattern1))
                    ckpt_dirs.update(search_path.glob(pattern2))
                    ckpt_dirs = sorted(ckpt_dirs)
                    
                    for ckpt_dir in ckpt_dirs:
                        # Check if experiment_data.json exists
                        if not (ckpt_dir / "experiment_data.json").exists():
                            continue
                        
                        # Check filter
                        if filter_pattern and filter_pattern not in str(ckpt_dir):
                            continue
                        
                        # Determine base model from model name
                        base_model = self._get_base_model_name(model)
                        
                        # 为每个patching_position_type创建一个job
                        for patching_position_type in patching_position_types:
                            # Create job name
                            job_name = f"{task}_{model}_{datasize}_{ckpt_dir.name}_{patching_position_type}"
                            
                            self.add_job(
                                name=job_name,
                                ckpt_dir=str(ckpt_dir),
                                task_name=task,
                                base_model=base_model,
                                patching_position_type=patching_position_type
                            )
        
        print(f"\nDiscovered {len(self.jobs)} jobs ({len(self.jobs)//len(patching_position_types)} checkpoints × {len(patching_position_types)} position types)")


    def _get_base_model_name(self, model_short_name: str) -> str:
        """Convert short model name to full HuggingFace model name."""
        model_mapping = {
            "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
            "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
            "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
            # Add more mappings as needed
        }
        return model_mapping.get(model_short_name, model_short_name)
    
    def load_jobs_from_manifest(self, manifest_path: str):
        """
        Load jobs from a YAML manifest file.
        
        Manifest format:
        experiments:
          - name: "exp1"
            ckpt_dir: "/path/to/checkpoint"
            task: "chaining"
            base_model: "Qwen/Qwen2.5-3B-Instruct"
          - name: "exp2"
            ...
        """
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        for exp in manifest.get("experiments", []):
            self.add_job(
                name=exp["name"],
                ckpt_dir=exp["ckpt_dir"],
                task_name=exp["task"],
                base_model=exp.get("base_model", "Qwen/Qwen2.5-3B-Instruct"),
                metric=exp.get("metric", "mrr"),
                epoch_offset=exp.get("epoch_offset", 0)
            )
    
    def _run_job(self, job: PatchingJob, gpu_id: int):
        """Run a single patching experiment job."""
        job.gpu_id = gpu_id
        job.status = "running"
        
        # Create log file
        log_file = self.log_dir / f"{job.name}.log"
        
        # Build command
        cmd = [
            sys.executable,
            self.script_path,
            "--ckpt_dir", job.ckpt_dir,
            "--base_model", job.base_model,
            "--task", job.task_name,
            "--device", "cuda:0",
            "--metric", job.metric,
            "--patching_position_type", job.patching_position_type  # 新增参数
        ]
        
        print(f"[GPU {gpu_id}] Starting: {job.name}")
        print(f"[GPU {gpu_id}] Task: {job.task_name}, Position type: {job.patching_position_type}, Checkpoint: {Path(job.ckpt_dir).name}")
        print(f"[GPU {gpu_id}] Log: {log_file}")
        
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
                print(f"[GPU {gpu_id}] ✓ Completed: {job.name}")
            else:
                job.status = "failed"
                job.error_msg = f"Exit code: {job.process.returncode}"
                print(f"[GPU {gpu_id}] ✗ Failed: {job.name} ({job.error_msg})")
                
        except Exception as e:
            job.status = "failed"
            job.error_msg = str(e)
            print(f"[GPU {gpu_id}] ✗ Error: {job.name}: {e}")
        
        finally:
            with self.lock:
                if gpu_id in self.running_jobs:
                    del self.running_jobs[gpu_id]
            self.available_gpus.put(gpu_id)
    
    def run_all(self):
        """Run all jobs, distributing across GPUs."""
        pending_jobs = [j for j in self.jobs if j.status == "pending"]
        skipped_jobs = [j for j in self.jobs if j.status == "skipped"]
        
        print(f"\n{'='*60}")
        print(f"Total jobs: {len(self.jobs)}")
        print(f"  Pending: {len(pending_jobs)}")
        print(f"  Skipped (already done): {len(skipped_jobs)}")
        print(f"Available GPUs: {self.gpu_ids}")
        print(f"{'='*60}\n")
        
        if not pending_jobs:
            print("No pending jobs to run!")
            self._print_summary()
            return
        
        threads = []
        job_index = 0
        
        while job_index < len(self.jobs) or threads:
            # Clean up completed threads
            threads = [t for t in threads if t.is_alive()]
            
            # Check stop flag
            if self.stop_flag:
                print("\nStop flag set, waiting for running jobs to complete...")
                for t in threads:
                    t.join()
                break
            
            # Start new jobs if GPUs available
            while job_index < len(self.jobs) and not self.available_gpus.empty():
                job = self.jobs[job_index]
                job_index += 1
                
                # Skip already completed/skipped jobs
                if job.status != "pending":
                    continue
                
                gpu_id = self.available_gpus.get()
                
                t = Thread(target=self._run_job, args=(job, gpu_id))
                t.start()
                threads.append(t)
                print(f"→ Submitted sample to GPU {gpu_id} ({len(self.jobs) - job_index} remaining)")
            
            time.sleep(1)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print job execution summary."""
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        completed = [j for j in self.jobs if j.status == "completed"]
        failed = [j for j in self.jobs if j.status == "failed"]
        pending = [j for j in self.jobs if j.status == "pending"]
        skipped = [j for j in self.jobs if j.status == "skipped"]
        
        print(f"Total: {len(self.jobs)}")
        print(f"  Completed: {len(completed)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Skipped (already done): {len(skipped)}")
        print(f"  Pending: {len(pending)}")
        
        # Group by task
        task_summary = {}
        for job in self.jobs:
            task = job.task_name
            if task not in task_summary:
                task_summary[task] = {
                    "completed": 0, "failed": 0, "skipped": 0, "pending": 0
                }
            task_summary[task][job.status] += 1
        
        print(f"\nBreakdown by task:")
        for task, stats in sorted(task_summary.items()):
            print(f"  {task}:")
            print(f"    Completed: {stats['completed']}, Failed: {stats['failed']}, "
                  f"Skipped: {stats['skipped']}, Pending: {stats['pending']}")
        
        if failed:
            print(f"\nFailed jobs:")
            for job in failed:
                error_info = f" ({job.error_msg})" if job.error_msg else ""
                print(f"  - {job.name}{error_info}")
        
        # Save summary
        summary = {
            "total": len(self.jobs),
            "completed": [{"name": j.name, "task": j.task_name, "ckpt_dir": j.ckpt_dir} 
                         for j in completed],
            "failed": [{"name": j.name, "task": j.task_name, "ckpt_dir": j.ckpt_dir,
                       "error": j.error_msg} for j in failed],
            "skipped": [{"name": j.name, "task": j.task_name, "ckpt_dir": j.ckpt_dir} 
                       for j in skipped],
            "pending": [{"name": j.name, "task": j.task_name, "ckpt_dir": j.ckpt_dir} 
                       for j in pending],
            "task_summary": task_summary
        }
        
        summary_path = self.log_dir / "summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        print(f"\nSummary saved to: {summary_path}")
    
    def stop(self):
        """Signal to stop scheduling new jobs."""
        self.stop_flag = True
        print("\nStopping scheduler (will wait for running jobs)...")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU patching experiment scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=str, help="Path to manifest YAML file")
    input_group.add_argument("--base_dir", type=str, 
                            help="Base directory to auto-discover checkpoints")
    
    # Auto-discovery parameters
    parser.add_argument("--models", type=str, nargs='+',
                       help="Model names (e.g., qwen2.5-3b qwen2.5-7b)")
    parser.add_argument("--tasks", type=str, nargs='+',
                       choices=['chaining', 'counting', 'intersection'],
                       help="Task names")
    parser.add_argument("--datasizes", type=str, nargs='+',
                       help="Data sizes (e.g., n1 n2 n3)")
    parser.add_argument("--filter", type=str, default=None,
                       help="Filter pattern for checkpoint names")
    parser.add_argument("--patching_position_types", type=str, nargs='+',  # 新增参数
                       default=["entity"],
                       help="Patching position types (e.g., entity bos eos random relation1 relation2)")
    
    # Execution parameters
    parser.add_argument("--gpus", type=str, required=True,
                       help="Comma-separated GPU IDs (e.g., 0,1,2,3)")
    parser.add_argument("--log_dir", type=str, default="./patching_logs",
                       help="Directory for log files")
    parser.add_argument("--script", type=str, 
                       default="multi_fact_experiment.py",
                       help="Path to patching experiment script")
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    
    # Create scheduler
    scheduler = GPUScheduler(gpu_ids, log_dir=args.log_dir, script_path=args.script)
    
    # Load jobs
    if args.manifest:
        print(f"Loading jobs from manifest: {args.manifest}")
        scheduler.load_jobs_from_manifest(args.manifest)
    
    elif args.base_dir:
        if not all([args.models, args.tasks, args.datasizes]):
            parser.error("--base_dir requires --models, --tasks, and --datasizes")
        
        print(f"Auto-discovering checkpoints from: {args.base_dir}")
        print(f"  Models: {args.models}")
        print(f"  Tasks: {args.tasks}")
        print(f"  Data sizes: {args.datasizes}")
        print(f"  Patching position types: {args.patching_position_types}")  # 新增输出
        if args.filter:
            print(f"  Filter: {args.filter}")
        
        scheduler.discover_checkpoints(
            base_dir=args.base_dir,
            models=args.models,
            tasks=args.tasks,
            datasizes=args.datasizes,
            patching_position_types=args.patching_position_types,  # 传入参数
            filter_pattern=args.filter
        )
    
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