import json
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tkinter import (
    Tk,
    Button,
    Label,
    filedialog,
    Frame,
    Canvas,
    Scrollbar,
    VERTICAL,
    Toplevel,
    LEFT,
    RIGHT,
    BOTH,
    Y,
    HORIZONTAL,
    VERTICAL,
    Scale,
    simpledialog,
)
import threading
import numpy as np
import os
import time
from PIL import Image, ImageTk
import imageio
import random


class Generator(nn.Module):
    def __init__(self, imageSize: int, latentDim: int, generatorChannels: int) -> None:
        super(Generator, self).__init__()

        self.initSize = imageSize // 16  # Initial size before upsampling

        self.l1 = nn.Sequential(
            nn.Linear(latentDim, generatorChannels * 8 * self.initSize**2)
        )

        self.convBlocks = nn.Sequential(
            nn.BatchNorm2d(generatorChannels * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                generatorChannels * 8, generatorChannels * 4, 3, stride=1, padding=1
            ),
            nn.BatchNorm2d(generatorChannels * 4, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                generatorChannels * 4, generatorChannels * 2, 3, stride=1, padding=1
            ),
            nn.BatchNorm2d(generatorChannels * 2, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(generatorChannels * 2, generatorChannels, 3, stride=1, padding=1),
            nn.BatchNorm2d(generatorChannels, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(generatorChannels, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.l1(z.view(z.size(0), -1))
        out = out.view(out.shape[0], -1, self.initSize, self.initSize)
        img = self.convBlocks(out)
        return img


class GanImageGenerator:
    def __init__(self):
        self.generator: Generator | None = None
        self.latentDim: int | None = None
        self.generatorChannels: int | None = None
        self.imageSize: int | None = None
        self.noiseVector: torch.Tensor | None = None

        self.root = Tk()
        self.root.title("GAN Image Generator")
        self.root.geometry("300x200")
        self.root.resizable(False, False)

        # Centering Frame for buttons
        buttonFrame = Frame(self.root)
        buttonFrame.pack(expand=True)

        # Load Model Button
        Button(buttonFrame, text="Load Model", command=self.loadModel).pack(pady=5)

        # Generate and Save Buttons
        Button(buttonFrame, text="Generate Random", command=self.generateRandom).pack(
            pady=5
        )
        Button(
            buttonFrame,
            text="Generate from Sliders",
            command=self.generateFromSliders,
        ).pack(pady=5)

        # Clear Images Button
        Button(
            buttonFrame, text="Clear Images", command=self.clearGeneratedImages
        ).pack(pady=5)

        # Generate Video Button
        Button(buttonFrame, text="Generate Video", command=self.generateVideo).pack(
            pady=5
        )

        # Slider window
        self.sliderWindow: Toplevel | None = None
        self.sliders: list[Scale] = []

        # Image display window
        self.imageWindow: Toplevel | None = None
        self.imageLabel: Label | None = None
        self.currentImage: Image.Image | None = None  # Store the current PIL Image

        # Ensure "generated" directory exists
        os.makedirs("generated", exist_ok=True)

    def loadModel(self):
        modelPath = filedialog.askdirectory(initialdir=os.path.join("", "models"))
        if modelPath:
            modelJsonPath = os.path.join(modelPath, "model.json")
            try:
                # Load the JSON file
                with open(modelJsonPath, "r") as f:
                    modelInfo = json.load(f)

                # Extract parameters from JSON
                loadInfo = modelInfo.get("loadInfo", {})
                self.latentDim = loadInfo.get("latent", 32)
                self.generatorChannels = loadInfo.get("channels", 136)
                self.imageSize = loadInfo.get("size", 128)

                # Initialize the generator model with the loaded parameters
                self.generator = Generator(
                    self.imageSize, self.latentDim, self.generatorChannels
                )

                # Load the state dictionary into the generator
                modelPath = os.path.join(modelPath, "generator.pth")
                stateDict = torch.load(modelPath, map_location="cpu")
                self.generator.load_state_dict(stateDict)
                self.generator.eval()
                print(
                    f"Model loaded successfully with latentDim={self.latentDim}, generatorChannels={self.generatorChannels}, imageSize={self.imageSize}"
                )

                self.createSliders(self.latentDim)
                self.noiseVector = torch.randn(1, self.latentDim, 1, 1)
                print(
                    f"Model loaded with latentDim={self.latentDim}, generatorChannels={self.generatorChannels}, imageSize={self.imageSize}"
                )

            except ValueError:
                print("Error: Please check the model parameters.")
            except RuntimeError as e:
                print(f"Error loading model state_dict: {e}")
            except FileNotFoundError:
                print("Model file not found.")
            except json.JSONDecodeError:
                print("Error decoding JSON file.")

    def createSliders(self, latentDim: int):
        sliderWidth = 220  # Width of each slider with padding
        numSlidersPerRow = 3  # Number of sliders per row
        windowHeight = 400  # Fixed height for the window
        windowSize = sliderWidth * numSlidersPerRow + 20

        # Create sliders in a separate window
        if self.sliderWindow:
            self.sliderWindow.destroy()
        self.sliderWindow = Toplevel(self.root)
        self.sliderWindow.title("Latent Space Sliders")
        self.sliderWindow.geometry(
            f"{windowSize + 30}x{windowHeight}"
        )  # Adjusted height
        self.sliderWindow.resizable(False, False)

        # Add a Canvas and Scrollbar for scrolling
        canvas = Canvas(self.sliderWindow, width=windowSize)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = Scrollbar(self.sliderWindow, orient=VERTICAL, command=canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)

        canvas.configure(yscrollcommand=scrollbar.set)

        sliderFrame = Frame(canvas)
        canvas.create_window((0, 0), window=sliderFrame, anchor="nw")

        self.sliders = []

        for i in range(latentDim):
            frame = Frame(sliderFrame)
            frame.grid(
                row=i // numSlidersPerRow,
                column=i % numSlidersPerRow,
                padx=10,
                pady=5,
            )

            label = Label(frame, text=f"Slider {i + 1}")
            label.pack(side="top")

            slider = Scale(
                frame,
                from_=-5,
                to=5,
                orient=HORIZONTAL,
                length=200,
                command=self.updateNoiseVector(i),
            )
            slider.set(0)
            slider.pack(side="top")
            self.sliders.append(slider)

        # Update the scroll region after all widgets are added
        sliderFrame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def updateNoiseVector(self, index: int):
        def inner(value: str):
            self.noiseVector[0, index, 0, 0] = float(value)

        return inner

    def generateRandom(self):
        if not self.generator:
            return
        self.noiseVector = torch.randn(1, len(self.sliders), 1, 1)
        for i, slider in enumerate(self.sliders):
            slider.set(self.noiseVector[0, i, 0, 0].item())
        self.generateImage()

    def generateFromSliders(self):
        if not self.generator:
            return
        self.generateImage()

    def generateImage(self):
        if not self.generator:
            return

        # Set up the video directory
        videoDir = "generated/images"
        os.makedirs(videoDir, exist_ok=True)
        with torch.no_grad():
            generatedImage = self.generator(self.noiseVector).detach().cpu()
            timestamp = int(time.time())
            fileName = f"generated/images/imagex{timestamp}.png"
            vutils.save_image(generatedImage, fileName, normalize=True)
            print(f"Generated image saved as {fileName}")
            self.currentImage = Image.open(fileName)  # Load image here for reuse
            self.displayImage()

    def displayImage(self):
        if self.imageWindow is None:
            self.imageWindow = Toplevel(self.root)
            self.imageWindow.title("Generated Image")
            self.imageWindow.geometry("512x512")  # Initial window size
            self.imageWindow.resizable(True, True)
            self.imageWindow.bind("<Configure>", self.resizeImage)  # Bind resize event
            self.imageLabel = Label(self.imageWindow)
            self.imageLabel.pack(expand=True)

        # Display the image with resizing
        self.resizeImage()

    def resizeImage(self, event=None):
        if self.currentImage:
            # Get the current window size
            window_width = self.imageWindow.winfo_width()
            window_height = self.imageWindow.winfo_height()

            # Calculate the aspect ratio
            aspect_ratio = self.currentImage.width / self.currentImage.height

            # Adjust the size while keeping the aspect ratio
            if window_width / window_height > aspect_ratio:
                new_height = window_height
                new_width = int(aspect_ratio * new_height)
            else:
                new_width = window_width
                new_height = int(new_width / aspect_ratio)

            # Resize the image to fit the new window size while maintaining aspect ratio
            img = self.currentImage.resize((new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage for Tkinter
            imgTk = ImageTk.PhotoImage(img)

            self.imageLabel.config(image=imgTk)
            self.imageLabel.image = imgTk

    def clearGeneratedImages(self):
        # Delete all files in the "generated" directory
        for fileName in os.listdir("generated"):
            filePath = os.path.join("generated", fileName)
            try:
                if os.path.isfile(filePath):
                    os.unlink(filePath)
                    print(f"Deleted {filePath}")
            except Exception as e:
                print(f"Failed to delete {filePath}. Reason: {e}")

    def generateVideo(self):
        if not self.generator:
            return

        # Prompt user for video length (in seconds) and FPS
        video_length = simpledialog.askinteger(
            "Video Length", "Enter video length (seconds):", minvalue=1, maxvalue=600
        )
        fps = simpledialog.askinteger(
            "FPS", "Enter frames per second (FPS):", minvalue=1, maxvalue=120
        )
        if video_length is None or fps is None:
            return  # If the user cancels the prompt

        # Calculate the number of frames based on video length and FPS
        num_frames = video_length * fps

        def _generate():

            # Set up the video directory
            videoDir = "generated/videos"
            os.makedirs(videoDir, exist_ok=True)

            # Store frames for video
            frames = []

            for frameIdx in range(num_frames):
                # Randomly change one slider's value
                randomIndex = torch.randint(0, len(self.sliders), (1,)).item()

                # Randomly decide whether to add or subtract 1
                change = random.choice([-1, 1])

                # Update the noise vector with clamping
                self.noiseVector[0, randomIndex, 0, 0] += change
                self.noiseVector[0, randomIndex, 0, 0] = torch.clamp(
                    self.noiseVector[0, randomIndex, 0, 0], -5, 5
                )

                # Update the corresponding slider
                self.sliders[randomIndex].set(
                    self.noiseVector[0, randomIndex, 0, 0].item()
                )

                # Generate the image
                with torch.no_grad():
                    generatedImage = self.generator(self.noiseVector).detach().cpu()
                    frame = generatedImage.squeeze().permute(1, 2, 0).numpy()
                    frame = ((frame + 1) * 127.5).astype(
                        np.uint8
                    )  # Rescale to [0, 255]
                    frames.append(frame)

                print(f"Generated frame {frameIdx + 1}/{num_frames}")

            # Use imageio to create a video
            timestamp = int(time.time())
            videoFileName = f"generated/videos/videox{timestamp}.mp4"
            os.makedirs(os.path.dirname(videoFileName), exist_ok=True)
            imageio.mimsave(videoFileName, frames, fps=fps)
            print(f"Video generated: {videoFileName}")

        thread = threading.Thread(target=_generate)
        thread.start()

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    ganImageGenerator = GanImageGenerator()
    ganImageGenerator.start()
