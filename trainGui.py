import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from threading import Thread, Event
from train import trainGan
import json
import os
import random
import platform


class GanTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GAN Trainer")

        # Define training parameters with default values
        self.params = {
            "numEpochs": tk.IntVar(value=random.randint(10, 500)),
            "batchSize": tk.IntVar(value=random.choice([8, 16, 24, 32, 48, 64])),
            "imageSize": tk.IntVar(value=random.choice([64, 128, 256, 512])),
            "latentDim": tk.IntVar(value=random.randint(16, 256)),
            "generatorChannels": tk.IntVar(value=random.randint(32, 512)),
            "discriminatorChannels": tk.IntVar(value=random.randint(32, 256)),
            "learningRate": tk.DoubleVar(value=0.0002),
            "beta1": tk.DoubleVar(value=0.6),
            "beta2": tk.DoubleVar(value=0.999),
            "lambdaGP": tk.IntVar(value=random.randint(5, 20)),
            "updateGFrequency": tk.IntVar(value=random.randint(2, 5)),
            "datasetFolder": tk.StringVar(value=""),
            "modelFolder": tk.StringVar(value=""),
            "genFolder": tk.StringVar(value=""),
        }

        # Dropdown for action on finish
        self.actionOnFinish = tk.StringVar(value="Nothing")

        # Store the input widgets for easy disabling/enabling
        self.inputWidgets = []

        # Add widgets for each parameter
        row = 0
        for key, var in self.params.items():
            label = ttk.Label(root, text=key)
            label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

            if key in ["datasetFolder", "modelFolder", "genFolder"]:
                entry = ttk.Entry(root, textvariable=var)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
                self.inputWidgets.append(entry)
                button = ttk.Button(
                    root, text="Browse", command=lambda k=key: self.selectFolder(k)
                )
                button.grid(row=row, column=2, padx=5, pady=5)
                self.inputWidgets.append(button)
            else:
                if isinstance(var, tk.StringVar):
                    entry = ttk.Entry(root, textvariable=var)
                else:
                    entry = ttk.Spinbox(
                        root, textvariable=var, from_=1, to=1000000, increment=1
                    )
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
                self.inputWidgets.append(entry)

            row += 1

        # Add dropdown for action on finish
        label = ttk.Label(root, text="Action on Finish")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        actionDropdown = ttk.Combobox(
            root,
            textvariable=self.actionOnFinish,
            values=["Nothing", "Shutdown", "Sleep"],
            state="readonly",
        )
        actionDropdown.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.inputWidgets.append(actionDropdown)

        row += 1

        # Add Start, Pause, Resume, and Stop buttons
        self.startButton = ttk.Button(
            root, text="Start Training", command=self.startTraining
        )
        self.startButton.grid(row=row, column=0, padx=5, pady=5)

        self.pauseButton = ttk.Button(
            root, text="Pause Training", command=self.pauseTraining, state="disabled"
        )
        self.pauseButton.grid(row=row, column=1, padx=5, pady=5)

        self.resumeButton = ttk.Button(
            root, text="Resume Training", command=self.resumeTraining, state="disabled"
        )
        self.resumeButton.grid(row=row + 1, column=0, padx=5, pady=5)

        self.stopButton = ttk.Button(
            root, text="Stop Training", command=self.stopTraining, state="disabled"
        )
        self.stopButton.grid(row=row + 1, column=1, padx=5, pady=5)

        self.trainingThread = None
        self.pauseEvent = Event()
        self.stopEvent = Event()

    def selectFolder(self, param):
        folder = filedialog.askdirectory()
        if folder:
            os.makedirs(folder, exist_ok=True)
            self.params[param].set(folder)

    def startTraining(self):
        if self.trainingThread is None or not self.trainingThread.is_alive():
            self.pauseEvent.clear()
            self.stopEvent.clear()
            self.trainingThread = Thread(target=self.runTraining)
            self.trainingThread.start()
            self.toggleInputs(state="disabled")
            self.startButton.config(state="disabled")
            self.pauseButton.config(state="normal")
            self.stopButton.config(state="normal")

    def pauseTraining(self):
        self.pauseEvent.set()
        self.pauseButton.config(state="disabled")
        self.resumeButton.config(state="normal")

    def resumeTraining(self):
        self.pauseEvent.clear()
        self.resumeButton.config(state="disabled")
        self.pauseButton.config(state="normal")

    def stopTraining(self):
        self.stopEvent.set()
        self.pauseEvent.set()  # Ensure paused training can be stopped
        self.trainingThread.join()
        self.trainingThread = None
        self.toggleInputs(state="normal")
        self.startButton.config(state="normal")
        self.pauseButton.config(state="disabled")
        self.resumeButton.config(state="disabled")
        self.stopButton.config(state="disabled")

    def toggleInputs(self, state):
        for widget in self.inputWidgets:
            widget.config(state=state)

    def readModelJson(self, modelFolder):
        """Read parameters from model.json if it exists."""
        jsonFile = os.path.join(modelFolder, "model.json")
        if os.path.exists(jsonFile):
            try:
                with open(jsonFile, "r") as f:
                    obj = json.load(f)
                    loadInfo = obj.get("loadInfo", {})

                    # Update parameters with values from the JSON file if they exist
                    if "latent" in loadInfo:
                        self.params["latentDim"].set(loadInfo["latent"])
                    if "channels" in loadInfo:
                        self.params["generatorChannels"].set(loadInfo["channels"])
                    if "size" in loadInfo:
                        self.params["imageSize"].set(loadInfo["size"])

                    print(f"Model parameters loaded from {jsonFile}")

            except Exception as e:
                print(f"Error reading {jsonFile}: {e}")

    def writeModelJson(self, modelFolder):
        """Write the current parameters to model.json."""
        jsonFile = os.path.join(modelFolder, "model.json")
        try:
            data = {
                "loadInfo": {
                    "latent": self.params["latentDim"].get(),
                    "channels": self.params["generatorChannels"].get(),
                    "size": self.params["imageSize"].get(),
                }
            }
            with open(jsonFile, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Model parameters saved to {jsonFile}")

        except Exception as e:
            print(f"Error writing {jsonFile}: {e}")

    def runTraining(self):
        modelFolder = self.params["modelFolder"].get()

        # Read parameters from model.json before starting the training
        self.readModelJson(modelFolder)

        # Unpack parameters and start the training
        try:
            # Write the current parameters to model.json
            self.writeModelJson(modelFolder)

            trainGan(
                numEpochs=self.params["numEpochs"].get(),
                batchSize=self.params["batchSize"].get(),
                imageSize=self.params["imageSize"].get(),
                latentDim=self.params["latentDim"].get(),
                generatorChannels=self.params["generatorChannels"].get(),
                discriminatorChannels=self.params["discriminatorChannels"].get(),
                learningRate=self.params["learningRate"].get(),
                beta1=self.params["beta1"].get(),
                beta2=self.params["beta2"].get(),
                lambdaGP=self.params["lambdaGP"].get(),
                updateGFrequency=self.params["updateGFrequency"].get(),
                datasetFolder=self.params["datasetFolder"].get(),
                modelFolder=self.params["modelFolder"].get(),
                genFolder=self.params["genFolder"].get(),
                pauseEvent=self.pauseEvent,
                stopEvent=self.stopEvent,
            )

            self.handleFinishAction()

        except Exception as e:
            print(f"Training error: {e}")

    def handleFinishAction(self):
        action = self.actionOnFinish.get()
        if action == "Shutdown":
            self.shutdown()
        elif action == "Sleep":
            self.sleep()

    def shutdown(self):
        if platform.system() == "Windows":
            os.system("shutdown /s /t 1")
        elif platform.system() == "Linux" or platform.system() == "Darwin":
            os.system("sudo shutdown -h now")

    def sleep(self):
        if platform.system() == "Windows":
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        elif platform.system() == "Linux":
            os.system("systemctl suspend")
        elif platform.system() == "Darwin":
            os.system("pmset sleepnow")


if __name__ == "__main__":
    root = tk.Tk()
    app = GanTrainerGUI(root)
    root.mainloop()
