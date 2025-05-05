import os
import shutil
import yaml
from pathlib import Path
import re

def parse_folder_name(folder_name):
    """Parse the folder name to extract configuration parameters."""
    pattern = r"(\d+)_sat_(\d+)_ant_(\d+)_usr_(\d+)_dist_([\d.]+)_error_on_cos_([\d.]+)_fading"
    match = re.match(pattern, folder_name)
    if match:
        return {
            'sat': int(match.group(1)),
            'ant': int(match.group(2)),
            'usr': int(match.group(3)),
            'dist': int(match.group(4)),
            'error_dist': float(match.group(5)),
            'error_cos': float(match.group(6))
        }
    return None

def reorganize_models(old_models_dir, new_models_dir):
    # Create new models directory if it doesn't exist
    new_models_dir.mkdir(parents=True, exist_ok=True)

    # Verify directories exist
    if not old_models_dir.exists():
        print(f"Error: {old_models_dir} does not exist!")
        return
    
    if not new_models_dir.exists():
        print(f"Error: {new_models_dir} does not exist!")
        return
    
    # Get all model folders from the old directory
    model_folders = [f for f in os.listdir(old_models_dir) 
                    if os.path.isdir(old_models_dir / f)]
    
    # Sort folders to ensure consistent numbering
    model_folders.sort()
    
    # Process each folder
    for idx, folder in enumerate(model_folders, 1):
        old_path = old_models_dir / folder
        new_path = new_models_dir / str(idx)
        
        # Parse configuration from folder name
        config = parse_folder_name(folder)
        if not config:
            print(f"Warning: Could not parse folder name: {folder}")
            continue
        
        # Create new folder
        os.makedirs(new_path, exist_ok=True)
        
        # Create config.yaml
        config_path = new_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Copy contents from old folder to new folder
        for item in os.listdir(old_path):
            s = old_path / item
            d = new_path / item
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        
        print(f"Processed {folder} -> {idx}")
    
    print("\nReorganization complete!")
    print(f"Models have been copied from: {old_models_dir}")
    print(f"to: {new_models_dir}")

if __name__ == "__main__":
    '''
    IMPORTANT: Change models folder name to models_old before running this script.
    '''
    project_root = Path(__file__).parent.parent.parent
    old_models_dir = project_root / 'models_old'
    new_models_dir = project_root / 'models'
    
    if not os.path.exists(old_models_dir):
        print(f"Error: Rename models folder to models_old before running this script.")
        exit()
    
    if os.path.exists(new_models_dir) and os.path.exists(old_models_dir):
        print(f"Error: models folder were already reformatted.")
        exit()

    reorganize_models() 