#!/usr/bin/env python3
"""
Batch training script for specific sample training.
Supports both single-fact and multi-fact training.
Randomly selects N sample IDs and runs training in parallel across multiple GPUs.

Usage:
    # Single-fact training
    python run_specific_samples.py --config configs/experiment_configs_specific_samples/specific_sample_template.yaml \
        --n_samples 100 --n_gpus 8
    
    # Multi-fact training
    python run_specific_samples.py --config configs/multi_experiment_configs_specific_samples/multi_specific_sample_template.yaml \
        --n_samples 50 --n_gpus 8
"""

import os
import sys
import random
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import json
import signal
from queue import Queue
from threading import Thread, Lock


class GPUScheduler:
    """Scheduler to distribute jobs across GPUs."""
    
    def __init__(self, gpu_ids: List[int], log_dir: str = "./experiment_logs_special_samples", training_script: str = "train_single_fact.py"):
        self.gpu_ids = gpu_ids
        self.available_gpus = Queue()
        for gpu_id in gpu_ids:
            self.available_gpus.put(gpu_id)
        
        self.jobs: List[Dict] = []
        self.results: List[Dict] = []
        self.running_jobs: Dict[int, Dict] = {}  # gpu_id -> job
        self.lock = Lock()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stop_flag = False
        self.config_path = None
        self.dry_run = False
        self.training_script = training_script  # NEW: Support both train_single_fact.py and train_multi_fact.py
        
    def add_job(self, sample_id: int):
        """Add a job to the queue."""
        job = {
            "sample_id": sample_id,
            "gpu_id": -1,
            "status": "pending",
            "success": False,
            "error": None,
            "duration": 0,
            "log_file": None
        }
        self.jobs.append(job)
    
    def _run_job(self, job: Dict, gpu_id: int):
        """Run a single training job."""
        sample_id = job["sample_id"]
        job["gpu_id"] = gpu_id
        job["status"] = "running"
        
        # Create log file
        log_file = self.log_dir / f"sample_{sample_id}_gpu_{gpu_id}.log"
        job["log_file"] = str(log_file)
        
        # Build command
        cmd = [
            sys.executable,
            self.training_script,  # Use detected training script
            "--config", self.config_path,
            "--specific_samples", str(sample_id),
        ]
        
        if self.dry_run:
            print(f"[DRY RUN] GPU {gpu_id}: Would run sample {sample_id} using {self.training_script}")
            job["status"] = "completed"
            job["success"] = True
            self.available_gpus.put(gpu_id)
            return
        
        print(f"[GPU {gpu_id}] Starting sample {sample_id} ({self.training_script})")
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                    cwd=os.path.dirname(os.path.abspath(__file__)) or "."
                )
            
            with self.lock:
                self.running_jobs[gpu_id] = job
                
            # Wait for completion
            process.wait()
            job["duration"] = time.time() - start_time
            
            if process.returncode == 0:
                job["status"] = "completed"
                job["success"] = True
                print(f"✓ [GPU {gpu_id}] Sample {sample_id} completed ({job['duration']:.1f}s)")
            else:
                job["status"] = "failed"
                # Read error from log file
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        job["error"] = log_content[-500:] if len(log_content) > 500 else log_content
                except:
                    pass
                print(f"✗ [GPU {gpu_id}] Sample {sample_id} failed (see {log_file})")
                
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["duration"] = time.time() - start_time
            print(f"✗ [GPU {gpu_id}] Sample {sample_id} exception: {e}")
        
        finally:
            with self.lock:
                if gpu_id in self.running_jobs:
                    del self.running_jobs[gpu_id]
            self.available_gpus.put(gpu_id)
    
    def run_all(self, config_path: str, dry_run: bool = False) -> List[Dict]:
        """Run all jobs, distributing across GPUs."""
        self.config_path = config_path
        self.dry_run = dry_run
        
        print(f"\n{'='*60}")
        print(f"Starting batch training")
        print(f"  Samples: {len(self.jobs)}")
        print(f"  GPUs: {self.gpu_ids}")
        print(f"  Config: {config_path}")
        print(f"  Script: {self.training_script}")
        print(f"  Log dir: {self.log_dir}")
        print(f"{'='*60}\n")
        
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
                
                print(f"→ Submitted sample {job['sample_id']} to GPU {gpu_id} ({len(self.jobs) - job_index} remaining)")
            
            time.sleep(0.5)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        return self.jobs
    
    def stop(self):
        """Signal to stop scheduling new jobs."""
        self.stop_flag = True
        print("\nStopping scheduler (will wait for running jobs)...")


def detect_training_type(config_path: str) -> tuple[str, Optional[str]]:
    """
    Detect training type (single-fact or multi-fact) from config.
    
    Returns:
        (training_script, task_type)
        - training_script: "train_single_fact.py" or "train_multi_fact.py"
        - task_type: None for single-fact, or task type for multi-fact
    """
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if it's multi-fact by looking for task_type field
        task_type = config.get('dataset', {}).get('task_type')
        if task_type and task_type.endswith('_tasks'):
            return "train_multi_fact.py", task_type
        
        # Default to single-fact
        return "train_single_fact.py", None
        
    except Exception as e:
        print(f"Warning: Could not detect training type from config: {e}")
        print("Defaulting to single-fact training")
        return "train_single_fact.py", None


def get_dataset_size(data_dir: str, config_path: str, task_type: Optional[str] = None) -> int:
    """
    Get the actual number of valid samples after deduplication and filtering.
    
    Args:
        data_dir: Path to dataset directory
        config_path: Path to config file
        task_type: For multi-fact, the task type (chaining_tasks, counting_tasks, intersection_tasks)
    
    Returns:
        Number of valid samples
    """
    import yaml
    import sys
    import io
    
    # Detect if single or multi fact
    _, detected_task_type = detect_training_type(config_path)
    task_type = task_type or detected_task_type
    
    try:
        # Temporarily suppress print statements
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            if task_type:
                # Multi-fact dataset
                from dataloader import MultiDataManager
                dataset = MultiDataManager(data_dir=data_dir)
                # For multi-fact, count the number of items in the specific task type
                actual_size = len(dataset.data.get(task_type, []))
            else:
                # Single-fact dataset
                from dataloader import SingleDataManager
                dataset = SingleDataManager(data_dir=data_dir)
                actual_size = len(dataset.data)
                
        finally:
            sys.stdout = old_stdout
        
        return actual_size
        
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch training for specific samples (single-fact or multi-fact)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to train")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g., 0,1,2,3), overrides --n_gpus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory from config")
    parser.add_argument("--sample_ids", type=str, default=None, 
                        help="Specific sample IDs to use (comma-separated), overrides random selection")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--output", type=str, default="batch_results.json", help="Output file for results")
    parser.add_argument("--log_dir", type=str, default="experiment_logs_special_samples", 
                        help="Directory for individual sample logs")
    parser.add_argument("--script", type=str, default=None,
                        choices=["train_single_fact.py", "train_multi_fact.py"],
                        help="Explicitly specify training script (auto-detect if not provided)")
    args = parser.parse_args()
    
    # Parse GPU IDs
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = list(range(args.n_gpus))
    print(f"Using GPUs: {gpu_ids}")
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = args.data_dir or config["dataset"]["data_dir"]
    
    # Detect training type
    if args.script:
        training_script = args.script
        task_type = config.get("dataset", {}).get("task_type")  # May be None for single-fact
        print(f"Training script: {training_script} (explicitly specified)")
    else:
        training_script, task_type = detect_training_type(args.config)
        print(f"Training script: {training_script} (auto-detected)")
    
    if task_type:
        print(f"Multi-fact task type: {task_type}")
    
    # Determine sample IDs
    if args.sample_ids:
        sample_ids = [int(x.strip()) for x in args.sample_ids.split(",")]
        print(f"Using specified sample IDs: {sample_ids}")
    else:
        # Get ACTUAL dataset size
        print("Loading dataset to determine actual size after filtering...")
        dataset_size = get_dataset_size(data_dir, args.config, task_type)
        
        if dataset_size == 0:
            print(f"Error: No valid samples found in {data_dir}")
            if task_type:
                print(f"Task type: {task_type}")
            print("Please check your dataset or specify --sample_ids")
            sys.exit(1)
        
        print(f"Actual available samples: {dataset_size}")
        if task_type:
            print(f"  (Note: Multi-fact - each sample contains 2 memorization tasks)")
        
        random.seed(args.seed)
        sample_ids = random.sample(range(dataset_size), min(args.n_samples, dataset_size))
        sample_ids.sort()
        print(f"Randomly selected {len(sample_ids)} samples")
        print(f"Sample IDs: {sample_ids[:10]}{'...' if len(sample_ids) > 10 else ''}")
    
    # Save selected sample IDs
    sample_ids_file = args.output.replace(".json", "_sample_ids.json")
    with open(sample_ids_file, 'w') as f:
        json.dump({
            "seed": args.seed, 
            "sample_ids": sample_ids,
            "training_script": training_script,
            "task_type": task_type
        }, f, indent=2)
    print(f"Sample IDs saved to: {sample_ids_file}")
    
    # Create scheduler
    scheduler = GPUScheduler(gpu_ids, log_dir=args.log_dir, training_script=training_script)
    
    # Add jobs
    for sample_id in sample_ids:
        scheduler.add_job(sample_id)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        scheduler.stop()
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run batch training
    start_time = time.time()
    results = scheduler.run_all(config_path=args.config, dry_run=args.dry_run)
    total_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"Batch Training Complete")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}")
    print(f"  Training script: {training_script}")
    if task_type:
        print(f"  Task type: {task_type}")
    print(f"  Logs directory: {scheduler.log_dir}")
    print(f"{'='*60}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            "config": args.config,
            "training_script": training_script,
            "task_type": task_type,
            "n_samples": len(sample_ids),
            "gpu_ids": gpu_ids,
            "seed": args.seed,
            "log_dir": str(scheduler.log_dir),
            "total_time": total_time,
            "successful": successful,
            "failed": failed,
            "results": results
        }, f, indent=2)
    print(f"Results saved to: {args.output}")
    
    # Print failed samples
    if failed > 0:
        print("\nFailed samples:")
        for r in results:
            if not r["success"]:
                error_preview = (r.get('error') or 'Unknown error')[:100]
                print(f"  Sample {r['sample_id']}: {error_preview}")
                print(f"    Log: {r.get('log_file', 'N/A')}")


if __name__ == "__main__":
    main()