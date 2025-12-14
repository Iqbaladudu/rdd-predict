
# Set your Cloudinary credentials
# ==============================
from dotenv import load_dotenv
load_dotenv()

# Import the Cloudinary libraries
# ==============================
import cloudinary
from cloudinary import CloudinaryImage
import cloudinary.uploader
import cloudinary.api

# Import to format the JSON responses
# ==============================
import json

# Set configuration parameter: return "https" URLs by setting secure=True  
# ==============================
config = cloudinary.config(secure=True)

# Log the configuration
# ==============================
print("****1. Set up and configure the SDK:****\nCredentials: ", config.cloud_name, config.api_key, "\n")

def upload_to_cloudinary(file_path, folder="rdd-predict"):
    """
    Upload a file to Cloudinary.
    
    Args:
        file_path: Path to the file to upload
        folder: Cloudinary folder name (default: "rdd-predict")
        
    Returns:
        dict: Upload result containing secure_url and other metadata
    """
    try:
        # Determine resource type based on file extension
        extension = file_path.split(".")[-1].lower()
        resource_type = "video" if extension in ["mp4", "avi", "mov", "mkv", "webm"] else "image"
        
        result = cloudinary.uploader.upload(
            file_path,
            folder=folder,
            resource_type=resource_type,
            overwrite=True
        )

        
        return {
            "url": result.get("secure_url"),
            "public_id": result.get("public_id"),
            "format": result.get("format"),
            "resource_type": result.get("resource_type")
        }
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        raise e
