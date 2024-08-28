import os
from tkinter import filedialog
from PIL import Image


def getTargetResolution():
    while True:
        try:
            width = int(input("Enter the target width: "))
            height = int(input("Enter the target height: "))
            return width, height
        except ValueError:
            print("Please enter valid integers for width and height.")


def cropImageCentered(image, targetWidth, targetHeight):
    imgWidth, imgHeight = image.size

    # Calculate coordinates for the crop box
    left = (imgWidth - targetWidth) / 2
    top = (imgHeight - targetHeight) / 2
    right = (imgWidth + targetWidth) / 2
    bottom = (imgHeight + targetHeight) / 2

    # Crop and return the image
    return image.crop((left, top, right, bottom))


def processImagesInDirectory(directoryPath, targetWidth, targetHeight):
    for fileName in os.listdir(directoryPath):
        filePath = os.path.join(directoryPath, fileName)

        if os.path.isfile(filePath) and fileName.lower().endswith(
            ("png", "jpg", "jpeg", "bmp", "gif")
        ):
            with Image.open(filePath) as img:
                croppedImage = cropImageCentered(img, targetWidth, targetHeight)

                # Save the cropped image (overwriting the original)
                croppedImage.save(filePath)
                print(f"Cropped and saved: {fileName}")


def main():
    # Ask user for the directory
    directoryPath = filedialog.askdirectory(title="Select Directory with Images")

    if not directoryPath:
        print("No directory selected. Exiting.")
        return

    # Get target resolution from the user
    targetWidth, targetHeight = getTargetResolution()

    # Process images in the directory
    processImagesInDirectory(directoryPath, targetWidth, targetHeight)


if __name__ == "__main__":
    main()
