"""
networks.domain_adaptation_schedules
====================================

Different learning rate schedules for the domain adaptation layers.
"""

import math
import torch.optim.lr_scheduler


class IncreasingPada(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    """
    The learning rate schedule from the PADA paper.

    See https://github.com/thuml/PADA. The math is not in the paper. Look in network.py on line 22:
    https://github.com/thuml/PADA/blob/534fb6c37b241fcf29b0ff15719649414794fa7f/pytorch/src/network.py#L22.
    This version does not include the gradient reversal built in to the PADA implementation.
    """

    def __init__(
        self,
        optimizer,
        lr_min: float,
        lambda_: float,
        alpha: float,
        max_epochs: float,
    ):
        self.__lr_min = lr_min
        self.__lambda = lambda_
        self.__alpha = alpha
        self.__max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__pada(base_lr) for base_lr in self.base_lrs]

    def __pada(self, base_lr: float) -> float:
        return (
            self.__lambda
            * (
                2.0
                * (base_lr - self.__lr_min)
                / (
                    1.0
                    + math.e ** (-self.__alpha * self.last_epoch / self.__max_epochs)
                )
            )
            - (base_lr - self.__lr_min)
            + self.__lr_min
        )


class IncreasingLinear(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    r"""
    A linear schedule that increases the learning rate.

    .. math::
        \begin{equation}
            \eta_t = \left(\frac{T_{cur}}{T_\max}\right)
                \left(\eta_\max - \eta_\min\right) + \eta_\min
        \end{equation}

    """

    def __init__(self, optimizer, lr_min: float, max_epochs: float):
        self.__lr_min = lr_min
        self.__max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__linear(base_lr) for base_lr in self.base_lrs]

    def __linear(self, base_lr: float) -> float:
        return (self.last_epoch / self.__max_epochs) * (
            base_lr - self.__lr_min
        ) + self.__lr_min


class IncreasingArccos(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    """
    A schedule using the arccos function in an increasing trend.

    This flips the cosine annealing function about the diagonal y=x (roughly speaking).
    """

    def __init__(self, optimizer, lr_min: float, max_epochs: int):
        self.__lr_min = lr_min
        self.__max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__arccos(base_lr) for base_lr in self.base_lrs]

    def __arccos(self, base_lr: float) -> float:
        return (
            -math.acos(2.0 * self.last_epoch / self.__max_epochs - 1.0) / math.pi + 1.0
        ) * (base_lr - self.__lr_min) + self.__lr_min


class IncreasingCosineAnnealing(torch.optim.lr_scheduler.CosineAnnealingLR):
    """
    A cosine annealing schedule that increases the learing rate.
    """

    # pylint: disable=useless-super-delegation
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        rates = super().get_lr()
        return [
            base_lr + self.eta_min - rate for base_lr, rate in zip(self.base_lrs, rates)
        ]

    def _get_closed_form_lr(self):
        rates = super()._get_closed_form_lr()  # pylint: disable=no-member
        return [
            base_lr + self.eta_min - rate for base_lr, rate in zip(self.base_lrs, rates)
        ]


class IncreasingExponential(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    """An exponential schedule that follows an increasing trend. """

    def __init__(self, optimizer, gamma: float):
        self.__gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__exponential(base_lr) for base_lr in self.base_lrs]

    def __exponential(self, base_lr: float) -> float:
        return -base_lr * self.__gamma ** self.last_epoch + base_lr


class IncreasingGamma(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    r"""
    A gamma function schedule that increases the learning rate.

    .. math::
        \begin{equation}
            \eta_t = \left(\frac{T_{cur}}{T_\max}\right)^{\frac{1}{\gamma}}
                \left(\eta_\max - \eta_\min\right) + \eta_\min
        \end{equation}

    """

    def __init__(self, optimizer, lr_min: float, gamma: float, max_epochs: float):
        self.__lr_min = lr_min
        self.__exponent = 1.0 / gamma
        self.__max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__gamma(base_lr) for base_lr in self.base_lrs]

    def __gamma(self, base_lr: float) -> float:
        return (
            math.pow(self.last_epoch / self.__max_epochs, self.__exponent)
            * (base_lr - self.__lr_min)
            + self.__lr_min
        )


class DecreasingPada(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    """
    The learning rate schedule from the PADA paper.

    See https://github.com/thuml/PADA. The math is not in the paper. Look in network.py on line 22:
    https://github.com/thuml/PADA/blob/534fb6c37b241fcf29b0ff15719649414794fa7f/pytorch/src/network.py#L22.
    This version does not include the gradient reversal built in to the PADA implementation.
    """

    def __init__(
        self,
        optimizer,
        lr_min,
        lambda_: float,
        alpha: float,
        max_epochs: float,
    ):
        self.__lr_min = lr_min
        self.__lambda = lambda_
        self.__alpha = alpha
        self.__max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__pada(base_lr) for base_lr in self.base_lrs]

    def __pada(self, base_lr) -> float:
        return (
            self.__lambda
            * (2.0 * (base_lr - self.__lr_min))
            / (1.0 + math.e ** (self.__alpha * self.last_epoch / self.__max_epochs))
            + self.__lr_min
        )


class DecreasingLinear(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    """
    A schedule that is a straight line down from the base learning rate to the minimum learning
    rate.
    """

    def __init__(self, optimizer, min_lr: float, max_epochs: int):
        self.__min_lr = min_lr
        self.__max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__linear(base_lr) for base_lr in self.base_lrs]

    def __linear(self, base_lr: float) -> float:
        return (-self.last_epoch / self.__max_epochs + 1.0) * (
            base_lr - self.__min_lr
        ) + self.__min_lr


class DecreasingGamma(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    r"""
    A gamma function schedule that decreases the learning rate.

    .. math::
        \begin{equation}
            \eta_t = -\left(\frac{T_{cur}}{T_\max}\right)^{\frac{1}{\gamma}}
                \left(\eta_\max - \eta_\min\right) + \eta_\max
        \end{equation}

    """

    def __init__(self, optimizer, lr_min: float, gamma: float, max_epochs: float):
        self.__lr_min = lr_min
        self.__exponent = 1.0 / gamma
        self.__max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__gamma(base_lr) for base_lr in self.base_lrs]

    def __gamma(self, base_lr: float) -> float:
        return (
            -math.pow(self.last_epoch / self.__max_epochs, self.__exponent)
            * (base_lr - self.__lr_min)
            + base_lr
        )


class DecreasingArccos(
    torch.optim.lr_scheduler._LRScheduler
):  # pylint: disable=protected-access
    """
    A schedule using the arccos function in a decreasing trend.

    This flips the cosine annealing function about the diagonal y=x (roughly speaking).
    """

    def __init__(self, optimizer, lr_min: float, max_epochs: int):
        self.__lr_min = lr_min
        self.__max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        return [self.__arccos(base_lr) for base_lr in self.base_lrs]

    def __arccos(self, base_lr: float) -> float:
        return (
            math.acos((2.0 * self.last_epoch) / self.__max_epochs - 1.0)
            / math.pi
            * (base_lr - self.__lr_min)
            + self.__lr_min
        )


def make_schedule(optimizer, direction: str, configuration):
    """
    Make a learning rate schedule.

    :param str schedule: The name of the learning rate schedule to create.
    :param optimizer: The Torch optimizer that will be adjusted by the schedule.
    :param parameters: The parameters to pass on to the schedule's __init__() method.
    :returns: The requested schedule.
    :raises ValueError: if ``schedule`` does not name a known schedule.
    """
    if configuration["schedule"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    if direction == "increasing":
        return _make_increasing_schedule(optimizer, configuration)
    if direction == "decreasing":
        return _make_decreasing_schedule(optimizer, configuration)
    raise ValueError("The schedule is not constant and the direction is invalid.")


def _make_increasing_schedule(optimizer, parameters):
    if parameters["schedule"] == "pada":
        return IncreasingPada(
            optimizer,
            parameters["minimum_rate"],
            parameters["pada"]["lambda"],
            parameters["pada"]["alpha"],
            parameters["maxiter_update"],
        )
    if parameters["schedule"] == "linear":
        return IncreasingLinear(
            optimizer, parameters["minimum_rate"], parameters["maxiter_update"]
        )
    if parameters["schedule"] == "cosine_annealing":
        return IncreasingCosineAnnealing(
            optimizer, parameters["maxiter_update"], parameters["minimum_rate"]
        )
    if parameters["schedule"] == "gamma":
        return IncreasingGamma(
            optimizer,
            parameters["minimum_rate"],
            parameters["gamma"]["gamma"],
            parameters["maxiter_update"],
        )
    if parameters["schedule"] == "exponential":
        return IncreasingExponential(optimizer, parameters["exponential"]["gamma"])
    if parameters["schedule"] == "arccosine":
        return IncreasingArccos(
            optimizer, parameters["minimum_rate"], parameters["maxiter_update"]
        )
    raise ValueError("Unknown schedule requested: " + parameters["schedule"])


def _make_decreasing_schedule(optimizer, parameters):
    if parameters["schedule"] == "pada":
        return DecreasingPada(
            optimizer,
            parameters["minimum_rate"],
            parameters["pada"]["lambda"],
            parameters["pada"]["alpha"],
            parameters["maxiter_update"],
        )
    if parameters["schedule"] == "linear":
        return DecreasingLinear(
            optimizer, parameters["minimum_rate"], parameters["maxiter_update"]
        )
    if parameters["schedule"] == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            parameters["maxiter_update"],
            parameters["minimum_rate"],
        )
    if parameters["schedule"] == "gamma":
        return DecreasingGamma(
            optimizer,
            parameters["minimum_rate"],
            parameters["gamma"]["gamma"],
            parameters["maxiter_update"],
        )
    if parameters["schedule"] == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, parameters["exponential"]["gamma"]
        )
    if parameters["schedule"] == "arccosine":
        return DecreasingArccos(
            optimizer, parameters["minimum_rate"], parameters["maxiter_update"]
        )
    raise ValueError("Unknown schedule requested: " + parameters["schedule"])
