from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import sys
from torchvision import transforms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--T', type=float, default=5, help='temperature parameter for knowledge distialation')
    return parser


class CRNew(ContinualModel):
    NAME = 'cr_new'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.learned_classes = 0

    def compute_pretext_task_loss(self, buf_outputs, buf_logits):
        # buf_outputs = buf_outputs[:, :self.learned_classes]
        # buf_logits = buf_logits[:, :self.learned_classes]
        if self.args.pretext_task == 'l1':
            loss = torch.pairwise_distance(buf_outputs, buf_logits, p=1).mean()
        elif self.args.pretext_task == 'l2':
            loss = torch.pairwise_distance(buf_outputs, buf_logits, p=2).mean()
        elif self.args.pretext_task == 'linf':
            loss = torch.pairwise_distance(buf_outputs, buf_logits, p=float('inf')).mean()
        elif self.args.pretext_task == 'kl':
            sim_logits = F.softmax(buf_logits)
            loss = F.kl_div(F.log_softmax(buf_outputs), sim_logits)
        elif self.args.pretext_task == 'kd':
            y_pred_soft = F.softmax(buf_outputs / self.args.T, dim=1)
            teacher_pred = F.softmax(buf_logits / self.args.T, dim=1)
            loss = F.kl_div(y_pred_soft, teacher_pred, size_average=False)
        elif self.args.pretext_task == 'cosine':
            loss = 1 - F.cosine_similarity(buf_outputs, buf_logits).mean()
        elif self.args.pretext_task == 'mse':
            loss = F.mse_loss(buf_outputs, buf_logits)
        else:
            raise ValueError("Unknown pretext task, got: ", self.args.pretext_task)

        return loss

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        # CE for current task samples
        loss = self.loss(outputs, labels)
        loss_1 = torch.tensor(0)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)

            # Pretext task
            loss_1 = self.compute_pretext_task_loss(buf_outputs, buf_logits)
            loss += self.args.alpha * loss_1

            # CE for buffered images
            buf_inputs_2, buf_labels_2, _ = self.buffer.get_data(
                self.args.batch_size, transform=self.transform)
            buf_outputs_2 = self.net(buf_inputs_2)
            loss += self.args.beta * self.loss(buf_outputs_2, buf_labels_2)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item(), loss_1.item()

    def update_logits(self):
        pass
        # with torch.no_grad():
        #     buf_inputs = self.buffer.examples
        #     buf_logits = self.buffer.logits
        #     buf_outputs = self.net(buf_inputs)
        #     new_logits = 0.7 * buf_logits + 0.3 * buf_outputs.data
        #     self.buffer.logits = new_logits
