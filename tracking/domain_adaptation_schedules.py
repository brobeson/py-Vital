"""
networks.domain_adaptation_schedules
====================================

Different learning rate schedules for the domain adaptation layers.
"""

import math
import torch.optim.lr_scheduler


class Constant:  # pylint: disable=too-few-public-methods
    r"""
    A constant schedule that increases the learning rate.

    .. math::
        \begin{equation}
            \eta_t = c
        \end{equation}

    """

    def __init__(self, constant: float):
        super().__init__()
        self.__constant = constant

    def __call__(self, epoch: int) -> float:
        return self.__constant


class IncreasingPada:  # pylint: disable=too-few-public-methods
    """
    The learning rate schedule from the PADA paper.

    See https://github.com/thuml/PADA. The math is not in the paper. Look in network.py on line 22:
    https://github.com/thuml/PADA/blob/534fb6c37b241fcf29b0ff15719649414794fa7f/pytorch/src/network.py#L22.
    This version does not include the gradient reversal built in to the PADA implementation.
    """

    def __init__(
        self, minimum_rate, maximum_rate, lambda_: float, alpha: float, epochs: float,
    ):
        self.__minimum_rate = minimum_rate
        self.__maximum_rate = maximum_rate
        self.__lambda = lambda_
        self.__alpha = alpha
        self.__number_of_epochs = epochs

    def __call__(self, epoch: int) -> float:
        """
        Calculate the learning rate for the given epoch.

        :param int epoch: The training epoch for which to calculate the learning rate.
        :return: The requested learning rate.
        :rtype: float
        """
        return (
            self.__lambda
            * (
                2.0
                * (self.__maximum_rate - self.__minimum_rate)
                / (1.0 + math.e ** (-self.__alpha * epoch / self.__number_of_epochs))
            )
            - (self.__maximum_rate - self.__minimum_rate)
            + self.__minimum_rate
        )


class IncreasingLinear:  # pylint: disable=too-few-public-methods
    r"""
    A linear schedule that increases the learning rate.

    .. math::
        \begin{equation}
            \eta_t = \left(\frac{T_{cur}}{T_\max}\right)
                \left(\eta_\max - \eta_\min\right) + \eta_\min
        \end{equation}

    """

    def __init__(self, minimum_rate, maximum_rate, epochs: float):
        super().__init__()
        self.__minimum_rate = minimum_rate
        self.__maximum_rate = maximum_rate
        self.__number_of_epochs = epochs

    def __call__(self, epoch: int) -> float:
        return (epoch / self.__number_of_epochs) * (
            self.__maximum_rate - self.__minimum_rate
        ) + self.__minimum_rate


# class IncreasingCosineAnnealing(torch.optim.lr_scheduler.CosineAnnealingLR):
#     """
#     A cosine annealing schedule that increases the learing rate.
#     """

#     # pylint: disable=useless-super-delegation
#     def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
#         super().__init__(optimizer, T_max, eta_min, last_epoch)

#     def get_lr(self):
#         rates = super().get_lr()
#         return [
#             base_lr + self.eta_min - rate for base_lr, rate in zip(self.base_lrs, rates)
#         ]

#     def _get_closed_form_lr(self):
#         rates = super()._get_closed_form_lr()  # pylint: disable=no-member
#         return [
#             base_lr + self.eta_min - rate for base_lr, rate in zip(self.base_lrs, rates)
#         ]


class IncreasingCosineAnnealing:  # pylint: disable=too-few-public-methods
    def __init__(self, minimum_rate: float, maximum_rate: float, epochs):
        self.__minimum_rete = minimum_rate
        self.__maximum_rate = maximum_rate
        self.__epochs = epochs

    def __call__(self, epoch):
        return self.__maximum_rate + 0.5 * (
            self.__minimum_rete - self.__maximum_rate
        ) * (1.0 + math.cos((math.pi * epoch) / self.__epochs))


class IncreasingGamma:  # pylint: disable=too-few-public-methods
    r"""
    A gamma function schedule that increases the learning rate.

    .. math::
        \begin{equation}
            \eta_t = \left(\frac{T_{cur}}{T_\max}\right)^{\frac{1}{\gamma}}
                \left(\eta_\max - \eta_\min\right) + \eta_\min
        \end{equation}

    """

    def __init__(self, minimum_rate, maximum_rate, gamma: float, epochs: float):
        super().__init__()
        self.__minimum_rate = minimum_rate
        self.__maximum_rate = maximum_rate
        self.__exponent = 1.0 / gamma
        self.__number_of_epochs = epochs

    def __call__(self, epoch: int) -> float:
        return (
            math.pow(epoch / self.__number_of_epochs, self.__exponent)
            * (self.__maximum_rate - self.__minimum_rate)
            + self.__minimum_rate
        )


class DecreasingPada:  # pylint: disable=too-few-public-methods
    """
    The learning rate schedule from the PADA paper.

    See https://github.com/thuml/PADA. The math is not in the paper. Look in network.py on line 22:
    https://github.com/thuml/PADA/blob/534fb6c37b241fcf29b0ff15719649414794fa7f/pytorch/src/network.py#L22.
    This version does not include the gradient reversal built in to the PADA implementation.
    """

    def __init__(
        self, minimum_rate, maximum_rate, lambda_: float, alpha: float, epochs: float,
    ):
        self.__minimum_rate = minimum_rate
        self.__maximum_rate = maximum_rate
        self.__lambda = lambda_
        self.__alpha = alpha
        self.__number_of_epochs = epochs

    def __call__(self, epoch: int) -> float:
        """
        Calculate the learning rate for the given epoch.

        :param int epoch: The training epoch for which to calculate the learning rate.
        :return: The requested learning rate.
        :rtype: float
        """
        return (
            self.__lambda
            * (
                2.0
                * (self.__maximum_rate - self.__minimum_rate)
                / (1.0 + math.e ** (self.__alpha * epoch / self.__number_of_epochs))
            )
            + self.__minimum_rate
        )


class DecreasingLinear:  # pylint: disable=too-few-public-methods
    r"""
    A linear schedule that decreases the learning rate.

    .. math::
        \begin{equation}
            \eta_t = \left(\frac{-T_{cur}}{T_\max} - 1\right)
                \left(\eta_\max - \eta_\min\right) + \eta_\min
        \end{equation}

    """

    def __init__(self, minimum_rate, maximum_rate, epochs: float):
        super().__init__()
        self.__minimum_rate = minimum_rate
        self.__maximum_rate = maximum_rate
        self.__number_of_epochs = epochs

    def __call__(self, epoch: int) -> float:
        return (-epoch / self.__number_of_epochs + 1.0) * (
            self.__maximum_rate - self.__minimum_rate
        ) + self.__minimum_rate


class DecreasingCosineAnnealing:  # pylint: disable=too-few-public-methods
    def __init__(self, minimum_rate: float, maximum_rate: float, epochs):
        self.__minimum_rete = minimum_rate
        self.__maximum_rate = maximum_rate
        self.__epochs = epochs

    def __call__(self, epoch):
        return self.__minimum_rete + 0.5 * (
            self.__maximum_rate - self.__minimum_rete
        ) * (1.0 + math.cos((math.pi * epoch) / self.__epochs))


class DecreasingGamma:  # pylint: disable=too-few-public-methods
    r"""
    A gamma function schedule that decreases the learning rate.

    .. math::
        \begin{equation}
            \eta_t = -\left(\frac{T_{cur}}{T_\max}\right)^{\frac{1}{\gamma}}
                \left(\eta_\max - \eta_\min\right) + \eta_\max
        \end{equation}

    """

    def __init__(self, minimum_rate, maximum_rate, gamma: float, epochs: float):
        super().__init__()
        self.__minimum_rate = minimum_rate
        self.__maximum_rate = maximum_rate
        self.__exponent = 1.0 / gamma
        self.__number_of_epochs = epochs

    def __call__(self, epoch: int) -> float:
        return (
            -math.pow(epoch / self.__number_of_epochs, self.__exponent)
            * (self.__maximum_rate - self.__minimum_rate)
            + self.__maximum_rate
        )


def make_schedule(optimizer, **parameters):
    """
    Make a learning rate schedule.

    :param str schedule: The name of the learning rate schedule to create.
    :param optimizer: The Torch optimizer that will be adjusted by the schedule.
    :param parameters: The parameters to pass on to the schedule's __init__() method.
    :returns: The requested schedule.
    :raises ValueError: if ``schedule`` does not name a known schedule.
    """
    if parameters["schedule"] == "constant":
        c = Constant(parameters["constant"])
        return torch.optim.lr_scheduler.LambdaLR(optimizer, c)
    if parameters["direction"] == "increasing":
        return _make_increasing_schedule(optimizer, parameters)
    if parameters["direction"] == "decreasing":
        return _make_decreasing_schedule(optimizer, parameters)
    raise ValueError("The schedule is not constant and the direction is invalid.")


def _make_increasing_schedule(optimizer, parameters):
    if parameters["schedule"] == "pada":
        pada = IncreasingPada(
            parameters["minimum_rate"],
            parameters["maximum_rate"],
            parameters["pada"]["lambda"],
            parameters["pada"]["alpha"],
            parameters["maxiter_update"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, pada)
    if parameters["schedule"] == "linear":
        linear = IncreasingLinear(
            parameters["minimum_rate"],
            parameters["maximum_rate"],
            parameters["maxiter_update"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, linear)
    if parameters["schedule"] == "cosine_annealing":
        c = IncreasingCosineAnnealing(
            parameters["minimum_rate"],
            parameters["maximum_rate"],
            parameters["maxiter_update"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, c)
    if parameters["schedule"] == "gamma":
        g = IncreasingGamma(
            parameters["minimum_rate"],
            parameters["maximum_rate"],
            parameters["gamma"],
            parameters["maxiter_update"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, g)
    raise ValueError("Unknown schedule requested: " + parameters["schedule"])


def _make_decreasing_schedule(optimizer, parameters):
    if parameters["schedule"] == "pada":
        pada = DecreasingPada(
            parameters["minimum_rate"],
            parameters["maximum_rate"],
            parameters["pada"]["lambda"],
            parameters["pada"]["alpha"],
            parameters["maxiter_update"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, pada)
    if parameters["schedule"] == "linear":
        linear = DecreasingLinear(
            parameters["minimum_rate"],
            parameters["maximum_rate"],
            parameters["maxiter_update"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, linear)
    if parameters["schedule"] == "cosine_annealing":
        c = DecreasingCosineAnnealing(
            parameters["minimum_rate"],
            parameters["maximum_rate"],
            parameters["maxiter_update"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, c)
    if parameters["schedule"] == "gamma":
        g = DecreasingGamma(
            parameters["minimum_rate"],
            parameters["maximum_rate"],
            parameters["gamma"],
            parameters["maxiter_update"],
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, g)
    raise ValueError("Unknown schedule requested: " + parameters["schedule"])
