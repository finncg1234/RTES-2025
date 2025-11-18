import subprocess
import itertools
from datetime import datetime

# Define parameter combinations
n_values = [64, 128]
b_values = [512]
r_values = [0.1, 0.5, 1]

# Fixed parameters
fixed_params = {
    'f': 2,
    'e': 500,
    'l': 1,
    'v': '2016-chevrolet-silverado',
    'd': 'combined'
}

# Generate all combinations
combinations = list(itertools.product(n_values, b_values, r_values))
total_runs = len(combinations)

print(f"Starting parameter sweep with {total_runs} combinations")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Track results
success_count = 0
failed_runs = []

for idx, (n, b, r) in enumerate(combinations, 1):
    print(f"[{idx}/{total_runs}] Running: n={n}, b={b}, r={r}")
    
    # Build command
    cmd = [
        'python', '.\\can_main.py',
        '-f', str(fixed_params['f']),
        '-b', str(b),
        '-r', str(r),
        '-e', str(fixed_params['e']),
        '-n', str(n),
        '-l', str(fixed_params['l']),
        '-v', fixed_params['v'],
        '-d', fixed_params['d'],
        '-run'
    ]
    
    try:
        # Execute command
        result = subprocess.run(cmd, check=True, text=True)
        print(f"  ✓ Success\n")
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed with return code {e.returncode}")
        print(f"  Error: {e.stderr[:200]}\n")
        failed_runs.append((n, b, r))
    except Exception as e:
        print(f"  ✗ Unexpected error: {str(e)}\n")
        failed_runs.append((n, b, r))

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total runs: {total_runs}")
print(f"Successful: {success_count}")
print(f"Failed: {len(failed_runs)}")

if failed_runs:
    print("\nFailed combinations:")
    for n, b, r in failed_runs:
        print(f"  n={n}, b={b}, r={r}")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")