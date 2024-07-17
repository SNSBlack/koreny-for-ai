import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

DATADIR = "edu_work" #это папка для заполнения файлов для обучения

INPUT_SIZE = 224

NUM_CLASSES = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ObjectDetector(nn.Module):
    def init(self, num_classes):
        super(ObjectDetector, self).init()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


model = ObjectDetector(NUM_CLASSES).to(DEVICE)


def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0).to(DEVICE)

    outputs = model(img)
    , preds = torch.max(outputs, 1)

    return preds


def getboundingbox(img, index, classid):
    return 0, 0, 100, 100


def trainmodel():


if __name__ == "__main__":
    train_model()


    for filename in os.listdir(DATA_DIR):                     #Фильтр короче я хзс как он будет точно работать
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(DATA_DIR, filename)
            preds = process_image(image_path)
        