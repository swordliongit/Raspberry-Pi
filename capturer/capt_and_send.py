import base64
import subprocess
import datetime
import os
import requests, json


def capture_image(output_path=None, rotation=0):
    """
    Capture an image using libcamera-still

    Args:
        output_path (str): Path where the image should be saved.
                          If None, creates a timestamped filename
        rotation (int): Rotation angle (0, 90, 180, or 270 degrees)

    Returns:
        str: Path to the saved image
    """
    # If no output path specified, create one with timestamp
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"capture_{timestamp}.jpg"

    # Construct the command
    command = ["libcamera-still", "-o", output_path, "--rotation", str(rotation)]

    try:
        # Run the capture command
        subprocess.run(command, check=True)
        print(f"Image captured successfully: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error capturing image: {e}")
        return None


def send_image():

    headers = {"Content-Type": "application/json"}
    payload = ""
    endpoint = "https://panel.xsarj.com/pr/send_img"

    with open("img.png", "rb") as file:
        # end open file
        img_bin = file.read()
        img_b64 = base64.b64encode(img_bin).decode("utf-8")
        payload = json.dumps({"image": img_b64})

    response = requests.post(url=endpoint, headers=headers, data=payload)

    print(response.text)


# Example usage
if __name__ == "__main__":
    # Basic capture
    # capture_image(output_path="img.png")
    send_image()

    # # Capture with custom filename and rotation
    # capture_image("my_picture.jpg", rotation=180)

    # # Capture multiple images with sequential naming
    # for i in range(3):
    #     capture_image(f"sequence_{i}.jpg")
