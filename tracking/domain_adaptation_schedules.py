"""
networks.domain_adaptation_schedules
====================================

Different learning rate schedules for the domain adaptation layers.
"""

import math
import torch.optim.lr_scheduler


class PadaScheduler:
    def __init__(
        self,
        max_epochs,
        lambda_: float,
        alpha: float,
    ):
        self.__lambda = lambda_
        self.__alpha = alpha
        self.__number_of_epochs = max_epochs

    def __call__(self, epoch: int) -> float:
        """
        Calculate the learning rate for the given epoch.

        :param int epoch: The training epoch for which to calculate the learning rate.
        :return: The requested learning rate.
        :rtype: float
        """
        # print("Running PADA scheduler.")
        return self.__lambda * (
            2.0 / (1.0 + math.e ** (-self.__alpha * epoch / self.__number_of_epochs))
            - 1
        )


class InverseCosineAnnealing(torch.optim.lr_scheduler.CosineAnnealingLR):
    # pylint: disable=useless-super-delegation
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        rates = super().get_lr()
        return [
            base_lr + self.eta_min - rate for base_lr, rate in zip(self.base_lrs, rates)
        ]

    def _get_closed_form_lr(self):
        rates = super()._get_closed_form_lr()
        return [
            base_lr + self.eta_min - rate for base_lr, rate in zip(self.base_lrs, rates)
        ]


# class GammaScheduler:
#     """Implements the gamma gradient reverse layer."""

#     def __init__(self, minimum_rate, maximum_rate, gamma: float, epochs: float):
#         super().__init__()
#         self.__minimum_rate = minimum_rate
#         self.__maximum_rate = maximum_rate
#         self.__gamma = gamma
#         self.__number_of_epochs = epochs
#         self.__current_epoch = 0

#     def forward(self, x):
#         """Feed forward through the layer."""
#         self.__current_epoch += 1
#         return x

#     def learning_rate(self):
#         return (
#             math.pow(self.__current_epoch / self.__number_of_epochs, 1.0 / self.__gamma)
#             * (self.__maximum_rate - self.__minimum_rate)
#             + self.__minimum_rate
#         )

#     def reset(self):
#         self.__current_epoch = 0


# class Linear:
#     """Implements an increasing linear learning rate schedule."""

#     def __init__(self, minimum_rate, maximum_rate, epochs: float):
#         super().__init__()
#         self.__minimum_rate = minimum_rate
#         self.__maximum_rate = maximum_rate
#         self.__number_of_epochs = epochs
#         self.__current_epoch = 0

#     def forward(self, x):
#         """Feed forward through the layer."""
#         self.__current_epoch += 1
#         return x

#     def learning_rate(self):
#         """Calculate the new learning rate."""
#         return (self.__current_epoch / self.__number_of_epochs) * (
#             self.__maximum_rate - self.__minimum_rate
#         ) + self.__minimum_rate

#     def reset(self):
#         self.__current_epoch = 0


def constant(_):
    """
    Maintain a constant learning rate.

    Pass this function to torch.optim.lr_scheduler.LambdaLR(). The initial
    learning rate is then used throughout the training process. This may seem
    pointless, but it allows a model to avoid having to check for a defined
    scheduler. Branching is a source of bugs.

    Example:
        >>> # This is more error prone: did you properly check all locations that scheduler.step()
        >>> # might be called?
        >>> if scheduler is not None:
        >>>     scheduler.step()
    """
    # print("running constant")
    return 1.0


# def make_schedule(scheduler: str, optimizer, **parameters):
def make_schedule(optimizer, **parameters):
    """
    Make a learning rate scheduler.

    :param str scheduler: The name of the learning rate scheduler to create.
    :param optimizer: The Torch optimizer that will be adjusted by the scheduler.
    :param parameters: The parameters to pass on to the scheduler's __init__() method.
    :returns: The requested scheduler.
    :raises ValueError: if ``scheduler`` does not name a known scheduler.
    """
    if parameters["scheduler"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, constant)
    if parameters["scheduler"] == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=parameters["maxiter_update"],
            eta_min=parameters["cosine_annealing"]["lr_min"],
        )
    if parameters["scheduler"] == "inverse_cosine_annealing":
        return InverseCosineAnnealing(
            optimizer=optimizer,
            T_max=parameters["maxiter_update"],
            eta_min=parameters["inverse_cosine_annealing"]["lr_min"],
        )
    if parameters["scheduler"] == "pada":
        pada = PadaScheduler(
            max_epochs=parameters["maxiter_update"],
            lambda_=parameters["lambda"],
            alpha=parameters["alpha"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, pada)
    raise ValueError("Unknown scheduler requested: " + parameters["scheduler"])
