import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import os
from tqdm import tqdm
from threading import Event


# Custom Dataset
class LiminalPoolsDataset(Dataset):
    def __init__(self, folder: str, imageSize: int) -> None:
        self.imageFiles = [
            os.path.join(folder, file)
            for file in os.listdir(folder)
            if file.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((imageSize, imageSize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.imageFiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.imageFiles[idx]).convert("RGB")
        return self.transform(image)


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


class Discriminator(nn.Module):
    def __init__(self, imageSize: int, discriminatorChannels: int) -> None:
        super(Discriminator, self).__init__()

        def discriminatorBlock(
            inFilters: int, outFilters: int, bn: bool = True
        ) -> nn.Sequential:
            block = [
                nn.Conv2d(inFilters, outFilters, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if bn:
                block.append(nn.BatchNorm2d(outFilters, 0.8))
            return nn.Sequential(*block)

        self.outputSize = imageSize // 16
        self.model = nn.Sequential(
            discriminatorBlock(3, discriminatorChannels, bn=False),
            discriminatorBlock(discriminatorChannels, discriminatorChannels * 2),
            discriminatorBlock(discriminatorChannels * 2, discriminatorChannels * 4),
            discriminatorBlock(discriminatorChannels * 4, discriminatorChannels * 8),
        )

        self.advLayer = nn.Sequential(
            nn.Linear(discriminatorChannels * 8 * self.outputSize**2, 1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.advLayer(out)
        return validity


def gradientPenalty(
    discriminator: nn.Module,
    realData: torch.Tensor,
    fakeData: torch.Tensor,
    device: torch.device,
    lambdaGP: float,
) -> torch.Tensor:
    batchSize = realData.size(0)
    alpha = torch.rand(batchSize, 1, 1, 1, device=device).expand_as(realData)
    interpolates = (alpha * realData + (1 - alpha) * fakeData).requires_grad_(True)
    discInterpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=discInterpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(discInterpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batchSize, -1)
    gradientPenaltyValue = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambdaGP * gradientPenaltyValue


# Initialize weights
def weightsInit(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def trainGan(
    numEpochs: int,
    batchSize: int,
    imageSize: int,
    latentDim: int,
    generatorChannels: int,
    discriminatorChannels: int,
    learningRate: float,
    beta1: float,
    beta2: float,
    lambdaGP: float,
    updateGFrequency: int,
    datasetFolder: str = "liminal_pools",
    modelFolder: str = "models",
    genFolder: str = "generated",
    pauseEvent: Event = None,
    stopEvent: Event = None,
) -> None:
    dataset = LiminalPoolsDataset(datasetFolder, imageSize)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(
        imageSize=imageSize, latentDim=latentDim, generatorChannels=generatorChannels
    ).to(device)
    netD = Discriminator(
        imageSize=imageSize, discriminatorChannels=discriminatorChannels
    ).to(device)

    os.makedirs(modelFolder, exist_ok=True)
    os.makedirs(genFolder, exist_ok=True)

    generatorPath = os.path.join(modelFolder, "generator.pth")
    discriminatorPath = os.path.join(modelFolder, "discriminator.pth")

    if os.path.exists(generatorPath) and os.path.exists(discriminatorPath):
        netG.load_state_dict(torch.load(generatorPath))
        netD.load_state_dict(torch.load(discriminatorPath))
        print(
            f"Model loaded: generatorPath={generatorPath}, discriminatorPath={discriminatorPath}"
        )
    else:
        netG.apply(weightsInit)
        netD.apply(weightsInit)

    optimizerD = optim.Adam(netD.parameters(), lr=learningRate, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=learningRate, betas=(beta1, beta2))

    for epoch in range(numEpochs):
        if stopEvent and stopEvent.is_set():
            print("Training stopped.")
            return

        for i, data in enumerate(tqdm(dataloader)):
            if stopEvent and stopEvent.is_set():
                print("Training stopped.")
                return

            if pauseEvent:
                while pauseEvent.is_set():
                    print("Training paused.")
                    pauseEvent.wait()  # Wait until the pauseEvent is cleared

            netD.zero_grad()
            realData = data.to(device)
            batchSize = realData.size(0)

            noise = torch.randn(batchSize, latentDim, 1, 1, device=device)
            fakeData = netG(noise)

            discReal = netD(realData).mean()
            discFake = netD(fakeData.detach()).mean()
            lossD = (
                discFake
                - discReal
                + gradientPenalty(netD, realData, fakeData, device, lambdaGP)
            )
            lossD.backward()
            optimizerD.step()

            if i % updateGFrequency == 0:
                netG.zero_grad()
                fakeData = netG(noise)
                lossG = -netD(fakeData).mean()
                lossG.backward()
                optimizerG.step()

        if stopEvent and stopEvent.is_set():
            print("Training stopped.")
            break

        with torch.no_grad():
            vutils.save_image(
                fakeData[:25],
                os.path.join(genFolder, f"epoch_{epoch+1}.png"),
                normalize=True,
                nrow=5,
            )
            print(f"New image saved: epoch={epoch + 1}")

        torch.save(netG.state_dict(), generatorPath)
        torch.save(netD.state_dict(), discriminatorPath)
        print(
            f"Model saved: epoch={epoch + 1}, generatorPath={generatorPath}, discriminatorPath={discriminatorPath}"
        )

    print("Training finished.")
