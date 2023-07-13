import torch
import numpy as np
from torch.autograd import Variable

from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient


class Evaluation:
    def __init__(self, data_loader, loss_function):
        """Expecting the trained model here as well as the test dataloader
        and the current training instance (to get the used loss function and
        other parameters possibly)."""
        self.loss_function = loss_function
        self.data_loader = data_loader
        self.canonizers = [SequentialMergeBatchNorm()]
        self.composite = EpsilonGammaBox(low=-3.0, high=3.0, canonizers=self.canonizers)

    def evaluate_digit(self, image, model):
        """this is probably not working yet, but at least I got started on it
        LRP is an importance visualization method, showing which pixels spoke
        for and against the prediction. this is supposed to return the
        prediction as well as the heatmap image if everything went right."""
        with Gradient(model=model, composite=self.composite) as attributor:
            out, relevance = attributor(image, torch.eye(1000)[[0]])
            return out, relevance

    def accuracy(self, model):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = Variable(data), Variable(target)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        return 100.0 * correct / len(self.data_loader.dataset)

    def evaluate_testset(self, model):
        """this should be working, just copy pasted from the evaluate file in
        the given project. It calculates accuracy, loss and accuracy per class
        for the validation dataset."""
        model.eval()
        losses = []
        correct = 0
        n_classes = len(np.unique(self.data_loader.dataset.targets))
        correct_classes = np.zeros(n_classes, dtype=np.int64)
        wrong_classes = np.zeros(n_classes, dtype=np.int64)
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = Variable(data), Variable(target)
                output = model(data)
                losses.append(self.loss_function(output, target).item())
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                preds = output.data.max(dim=1)[1].cpu().numpy().astype(np.int64)
                target = target.data.cpu().numpy().astype(np.int64)
                for label, pred in zip(target, preds):
                    if label == pred:
                        correct_classes[label] += 1
                    else:
                        wrong_classes[label] += 1
        eval_loss = float(np.mean(losses))

        assert correct_classes.sum() + wrong_classes.sum() == len(
            self.data_loader.dataset
        )
        return {
            "evaluation_loss": eval_loss,
            "evaluation_accuracy": 100.0 * correct / len(self.data_loader.dataset),
            "accuracy_per_class": 100.0
            * correct_classes
            / (correct_classes + wrong_classes),
        }
