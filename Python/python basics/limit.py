import math
import multiprocessing
import time

# Function to check if a number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Worker function for multiprocessing
def find_primes_in_range(start, end):
    primes = [n for n in range(start, end) if is_prime(n)]
    return primes

# Main function
if __name__ == "__main__":
    limit = 10**10  # Finding primes up to 10 million
    num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores
    print(f"Using {num_cores} CPU cores...")

    start_time = time.time()

    # Define the ranges for multiprocessing
    step = limit // num_cores
    ranges = [(i, i + step) for i in range(2, limit, step)]

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(find_primes_in_range, ranges)

    # Flatten the results into a single list
    all_primes = [prime for sublist in results for prime in sublist]

    end_time = time.time()

    print(f"Found {len(all_primes)} prime numbers up to {limit}.")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")