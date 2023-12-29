import json
import numpy as np

# Open the input file from args
file_name = 'camera_path.json'

# Open the input file
with open(file_name) as f:
    # Load the JSON data
    data = json.load(f)

def get_translated_matrix(transform_matrix: list, shift: float):
    # Convert transform_matrix to numpy array
    transform_matrix = np.array(transform_matrix)

    # Create shift matrix
    shift_matrix = np.array([[1.0, 0.0, 0.0, shift],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

    # Calculate the translated matrix
    return np.linalg.inv(shift_matrix @ np.linalg.inv(transform_matrix)).tolist()

for i in range(len(data['camera_path'])):
    t_matrix = data['camera_path'][i]["camera_to_world"]
    data['camera_path'][i]["camera_to_world"] = get_translated_matrix(transform_matrix=t_matrix, shift=-0.2)
    
    
# Write the modified JSON data to the destination file
with open('camera_path_left.json', mode='w') as f:
    json.dump(data, f, indent=4)