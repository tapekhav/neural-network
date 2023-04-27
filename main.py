from numpy import exp, array, random, dot
from typing import NoReturn
from enum import Enum


class Case(Enum):
    FEW = 10
    NORMAL = 10**4
    MANY = 10**6


class NeuralNetwork:
    def __init__(self):
        """
        Мы моделируем один нейрон входными с тремя входными подключениями и 1 выходным
        Мы назначаем случайные веса со значениями в диапазоне от -1 до 1
        и средним значением 0
        """
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    @staticmethod
    def sigmoid(x) -> int:
        return 1 / (1 + exp(-x))

    @staticmethod
    def derivative(x: int) -> int:
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_iterations) -> NoReturn:
        """
        Обучает нейронную сеть, используя предоставленный обучающий набор и корректирует веса.
        :param training_set_inputs: numpy массив входных значений для тренировочного набора
        :param training_set_outputs: numpy массив ожидаемых выходных значений для обучающего набора
        :param number_iterations: количество раз, чтобы обучить сеть, используя предоставленный обучающий набор
        """
        for _ in range(number_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output

            self.synaptic_weights += dot(training_set_inputs.T, error * self.derivative(output))

    def think(self, inputs) -> int:
        """
        Подает входные значения через нейронную сеть и возвращает выходное значение.
        :param inputs: numpy массив входных значений для передачи по сети
        :return: выходное значение нейронной сети
        """
        return self.sigmoid(dot(inputs, self.synaptic_weights))


def main():
    # Инициализируем нейронную сеть с одним нейроном
    neural_network = NeuralNetwork()

    print("Случайные начальные веса: ")
    print(*neural_network.synaptic_weights)

    # Обучающий набор
    # У нас есть 4 примера, каждый из них состоит из 3 входных значений
    # и одно выходное значение
    inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    outputs = array([[0, 1, 1, 0]]).T

    # Тренируем нейросеть, используя обучающий набор
    # Делаем это 10000 раз и каждый раз делаем небольшие корректировки
    neural_network.train(inputs, outputs, Case.NORMAL.value)

    print("Новые веса после тренировки: ")
    print(*neural_network.synaptic_weights)

    # Проверим нейросеть с новыми числами
    print("Рассмотрим новую ситуацию [1, 0, 0]: ")
    print(neural_network.think(array([1, 0, 0])))


if __name__ == "__main__":
    main()

