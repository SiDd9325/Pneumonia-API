    

import torch
from torchvision.models import densenet121, DenseNet121_Weights

class PneumoniaClassifier:
    def __init__(self, model_path):
        weights = DenseNet121_Weights.IMAGENET1K_V1
        self.model = densenet121(weights=weights)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, tensor):
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item()

def get_model(model_path):
    return PneumoniaClassifier(model_path)
