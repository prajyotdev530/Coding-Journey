import time
import multiprocessing

# Function to check if a number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Function to find primes in a given range
def find_primes_in_range(start, end):
    primes = [n for n in range(start, end) if is_prime(n)]
    return primes

# Function to test CPU with multi-processing
def cpu_benchmark():
    print("Starting CPU benchmark...")

    num_cores = multiprocessing.cpu_count()  # Get number of CPU cores
    print(f"Using {num_cores} CPU cores")

    # Define range for prime number calculation
    LIMIT = 50_000_00  # 50 million  # 100 million  # Increase this value # Adjust as needed
    step = LIMIT // num_cores

    start_time = time.time()

    # Create multiple processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        ranges = [(i * step, (i + 1) * step) for i in range(num_cores)]
        results = pool.starmap(find_primes_in_range, ranges)

    end_time = time.time()
    
    print(f"CPU Benchmark completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    cpu_benchmark()