import time
import multiprocessing

def compute_operations(dummy):
    """Perform heavy computations to fully utilize CPU cores."""
    count = 0
    start_time = time.time()
    while time.time() - start_time < 1:  # Run for 1 second
        for _ in range(10**7):  # Increase workload
            count += 1
    return count

def cpu_benchmark():
    """Measure total CPU calculations per second using all cores."""
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    # Run the computation in parallel on all CPU cores
    results = pool.map(compute_operations, [None] * num_cores)
    
    total_count = sum(results)
    print(f"⚡ Total CPU Speed: {total_count / 1e9:.2f} Billion calculations per second")

# Run CPU benchmark
cpu_benchmark()