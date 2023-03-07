# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import sys
from torchvision import transforms


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class CR(ContinualModel):
    NAME = 'cr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CR, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        if self.args.pretext_task == 'mae':
            self.l1_loss = torch.nn.L1Loss()

    def compute_pretext_task_loss(self, buf_outputs, buf_logits):
        if self.args.pretext_task == 'l1':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=1).mean()
        elif self.args.pretext_task == 'l2':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=2).mean()
        elif self.args.pretext_task == 'linf':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=float('inf')).mean()
        elif self.args.pretext_task == 'kl':
            sim_logits = F.softmax(buf_logits)
            loss = self.args.alpha * F.kl_div(F.log_softmax(buf_outputs), sim_logits)
        else:
            loss = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

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
            loss += loss_1

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
