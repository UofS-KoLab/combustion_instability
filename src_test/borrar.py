import os
import pandas as pd

def list_files_in_folder(folder_path):
    """
    List all files in the specified folder.

    :param folder_path: Path to the folder.
    :return: List of file names.
    """
    # List all files and directories in the specified folder
    all_items = os.listdir(folder_path)
    
    # Filter out directories to get only files
    files = [item.split(".csv")[0] for item in all_items if os.path.isfile(os.path.join(folder_path, item))]
    
    return files

fuel_type="JP8_HRJ"

# Example usage
folder_path = 'C:\\Users\\qpw475\\Documents\\combustion_instability\\data\\raw\\'+fuel_type # Replace with the path to your folder
files = list_files_in_folder(folder_path)
print("Files in folder:", files)

# Create a DataFrame
data = {
    'Name': files,
    'Stability': ['Stable'] * len(files)
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_csv_path = 'C:\\Users\\qpw475\\Documents\\combustion_instability\\data\\labels'  # Replace with the desired output path
output_csv_path=os.path.join(output_csv_path, f"{fuel_type}_label.csv")
df.to_csv(output_csv_path, index=False)
print(f"DataFrame saved to {output_csv_path}")