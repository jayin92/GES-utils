import os
import glob
from pathlib import Path

def rename_images(folder_path='.'):
    """
    Rename images from *_00XX.jpeg to frames_00XX.jpg
    
    Args:
        folder_path: Path to the folder containing images (default: current directory)
    """
    # Change to the target folder
    os.chdir(folder_path)
    
    # Find all .jpeg files in the folder
    jpeg_files = glob.glob('*_*.jpeg')
    
    renamed_count = 0
    
    for filename in jpeg_files:
        # Extract the number part (assuming format *_XXXX.jpeg)
        try:
            # Split by underscore and get the last part before extension
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                # Get the number part (remove .jpeg extension)
                number_str = parts[1].replace('.jpg', '')
                
                # Convert to integer to validate
                frame_number = int(number_str)
                
                # Only rename if number is between 0 and 3000
                if 0 <= frame_number <= 3000:
                    # Create new filename with zero-padding to match original format
                    new_filename = f'frame_{number_str}.jpg'
                    
                    # Rename the file
                    os.rename(filename, new_filename)
                    print(f'Renamed: {filename} -> {new_filename}')
                    renamed_count += 1
                else:
                    print(f'Skipped (out of range): {filename}')
        except ValueError:
            print(f'Skipped (invalid format): {filename}')
        except Exception as e:
            print(f'Error processing {filename}: {e}')
    
    print(f'\nTotal files renamed: {renamed_count}')

if __name__ == '__main__':
    # Specify your folder path here, or use current directory
    folder_path = '/project/jayinnn/GES-utils/tokyo_tower/images'  # Change this to your folder path, e.g., '/path/to/images'
    rename_images(folder_path)