import subprocess
import sys
import os

def generate_dataset(dataset_name, duration=120, cores=8):
    """Generate wind and puff data for a dataset"""
    print(f"Generating simulation data for {dataset_name}...")
    
    cmd = [
        sys.executable, 'sim_cli.py',
        '--duration', str(duration),
        '--cores', str(cores),
        '--dataset', dataset_name
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"Successfully generated {dataset_name}")
            return True
        else:
            print(f"Error generating {dataset_name}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout generating {dataset_name}")
        return False
    except Exception as e:
        print(f"Exception generating {dataset_name}: {e}")
        return False

def main():
    """Generate simulation data for common datasets"""
    datasets = [
        'constantx5b5',
        'noisy3x5b5',
        'noisy6x5b5', 
        'switch45x5b5'
    ]
    
    # Check if sim_cli.py exists
    if not os.path.exists('sim_cli.py'):
        print("Error: sim_cli.py not found. Make sure you're in the correct directory.")
        return
    
    for dataset in datasets:
        generate_dataset(dataset)

if __name__ == '__main__':
    main()
