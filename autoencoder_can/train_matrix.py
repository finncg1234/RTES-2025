import subprocess
import itertools
from datetime import datetime
import os
import re

# Define parameter combinations
n_values = [224, 192]
b_values = [256, 512, 1024]
r_values = [18, 20, 25]
d_values = ['combined', 'fuzzy', 'DoS', 'gear', 'interval', 'rpm', 'speed', 'standstill']
bfws = ['-nobfw', '']

run_only = True
# d_values = ['extra-attack-free']

# Fixed parameters
fixed_params = {
    'f': 1,
    'e': 1000,
    'v': '2016-chevrolet-silverado',
    # 'd': 'fuzzy'
}

# Generate all combinations
combinations = list(itertools.product(n_values, b_values, r_values, bfws))
total_runs = len(combinations)

print(f"Starting parameter sweep with {total_runs} combinations")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Track results
success_count = 0
failed_runs = []

# For running the model, get all the paths to the models
# Set your folder path
folder_path = r".\\out\\2016-chevrolet-silverado\\weights"  # change this to your folder

# List all files ending with .pth
pth_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
# pth_files_full = [os.path.join(folder_path, f) for f in pth_files]

# print(pth_files)

if run_only:
    for f in pth_files:
        # fe-<number>
        fe_match = re.search(r'fe-(\d+)', f)
        fe = int(fe_match.group(1)) if fe_match else None

        # b-<number>
        b_match = re.search(r'b-(\d+)', f)
        b = int(b_match.group(1)) if b_match else None

         # input-size-<number>
        n_match = re.search(r'input-size-(\d+)', f)
        n = int(n_match.group(1)) if n_match else None

        # epochs-<number>
        e_match = re.search(r'epochs-(\d+)', f)
        e = int(e_match.group(1)) if e_match else None

        # lr-<number> (handle underscores as decimal points)
        r_match = re.search(r'lr-([0-9_]+)', f)
        if r_match:
            r = float(r_match.group(1).replace('_', '.'))
        else:
            r = None

        bfw = '-nobfw' if 'no-bfw' in f else ''

        for d in d_values:
            run_cmd = [
                'python', '.\\can_main.py',
                '-f', str(fe),
                '-b', str(b),
                '-r', str(r * 10000),
                '-e', str(e),
                '-n', str(int(n / 2)),
                '-l', '1',
                '-v', str(fixed_params['v']),
                '-d', str(d),
                '-run',
                "-model", os.path.join(folder_path, f)
            ]
            if bfw:  # only append if non-empty
                run_cmd.append(bfw)
            result = subprocess.run(run_cmd, check=True, text=True)
            print(f"  ✓ Success\n")
            # print(str(run_cmd) + "\n")
else:
    for idx, (n, b, r, bfw) in enumerate(combinations, 1):
        print(f"[{idx}/{total_runs}] Running: n={n}, b={b}, r={r}, bfw={bfw}")

        # train command
        train_cmd = [
            'python', '.\\can_main.py',
            '-f', str(fixed_params['f']),
            '-b', str(b),
            '-r', str(r),
            '-e', str(fixed_params['e']),
            '-n', str(n),
            '-l', '0',
            '-v', fixed_params['v'],
            '-d', 'extra-attack-free',
            '-train'
        ]
        if bfw:  # only append if non-empty
            train_cmd.append(bfw)

        try:
            # Execute command
            result = subprocess.run(train_cmd, check=True, text=True)
            print(f"  ✓ Success\n")
            success_count += 1
            for d in d_values:
                run_cmd = [
                    'python', '.\\can_main.py',
                    '-f', str(fixed_params['f']),
                    '-b', str(b),
                    '-r', str(r),
                    '-e', str(fixed_params['e']),
                    '-n', str(n),
                    '-l', '1',
                    '-v', fixed_params['v'],
                    '-d', str(d),
                    '-run',
                ]
                if bfw:  # only append if non-empty
                    run_cmd.append(bfw)
                result = subprocess.run(run_cmd, check=True, text=True)
                print(f"  ✓ Success\n")
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