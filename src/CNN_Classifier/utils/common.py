import yaml
import os
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from typing import Any
import joblib, json
import base64

# Define a function to read YAML files and return a ConfigBox object
@ensure_annotations
def read_yaml(path_to_yaml: str) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (str): The file path to the YAML file."""
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError(f"Error converting YAML content to ConfigBox: {e}")
    except Exception as e:
        raise ValueError(f"Error reading YAML file: {e}")
# Define a function to create directories if they don't exist
@ensure_annotations
def create_directories(path_to_directories: list) -> None:
    """
    Creates directories if they do not exist.

    Args:
        path_to_directories (list): A list of directory paths to create."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
# Define a function to save a Python object as a JSON file
@ensure_annotations
def save_json(path: str, data: Any) -> None:
    """
    Saves a Python object as a JSON file.

    Args:
        path (str): The file path where the JSON file will be saved.
        data (Any): The Python object to be saved as JSON."""
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
# Define a function to load a Python object from a JSON file
@ensure_annotations
def load_json(path: str) -> Any:
    """
    Loads a Python object from a JSON file.

    Args:
        path (str): The file path to the JSON file."""
    with open(path, 'r') as json_file:
        return json.load(json_file)
# Define a function to save a Python object using joblib
@ensure_annotations
def save_object(path: str, obj: Any) -> None:
    """
    Saves a Python object using joblib.

    Args:
        path (str): The file path where the object will be saved.
        obj (Any): The Python object to be saved."""
    joblib.dump(obj, path)
# Define a function to load a Python object using joblib
@ensure_annotations
def load_object(path: str) -> Any:
    """
    Loads a Python object using joblib.

    Args:
        path (str): The file path to the object file."""
    return joblib.load(path)

# Define a function to get the size of a file in KB
@ensure_annotations
def get_size(path: str) -> int:
    """
    Gets the size of a file in KB.

    Args:
        path (str): The file path to the file."""
    return os.path.getsize(path)/1024
# Define a function to encode an image to a Base64 string at the UI terminal
@ensure_annotations
def encodeImageToBase64(image_path: str) -> str:
    """
    Encodes an image to a Base64 string.

    Args:
        image_path (str): The file path to the image."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
    
def decodeBase64ToImage(base64_string: str, output_path: str) -> None:
    """
    Decodes a Base64 string back to an image and saves it to the specified output path.

    Args:
        base64_string (str): The Base64 string to be decoded.
        output_path (str): The file path where the decoded image will be saved."""
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_string))
        image_file.close()
