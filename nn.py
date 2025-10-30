import numpy as np

np.random.seed(42)

class Dense_layer:
    def __init__(self, n_inputs, n_neurons, weights_regularizer=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.lw2 = weights_regularizer

    def forward(self, input):
        self.inputs = input
        self.output = np.dot(input, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.lw2:
            self.dweights += self.lw2 * 2 * self.weights

        self.dinputs = np.dot(dvalues, self.weights.T)


class Dropout:
    def __init__(self, rate=0):
        self.rate = 1 - rate

    def forward(self, input):
        self.inputs = input
        self.binary_mask = (
            np.random.binomial(1, self.rate, size=input.shape) / self.rate
        )
        self.output = input * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class ReLU_activation:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0


class Sigmoid_activation:
    def forward(self, input):
        self.inputs = input
        self.output = 1 / (1 + np.exp(-np.clip(input, -10, 10)))

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)


class Adam_optimzer:
    def __init__(
        self, learning_rate=0.0001, decay=0, beta_1=0.99, beta_2=0.9, epsilon=1e-7
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        self.current_learning_rate = self.learning_rate * (
            1 / (1 + self.decay * self.iterations)
        )

    def update_params(self, layer):
        if not hasattr(layer, "weight_momentums"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )

        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (
            layer.dweights**2
        )
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (
            layer.dbiases**2
        )

        weight_cache_corrected = layer.weight_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )
        bias_cache_corrected = layer.bias_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


class Loss:
    def regularization_loss(self, layer):
        reg_loss = 0

        if layer.lw2:
            reg_loss += layer.lw2 * np.sum(layer.weights**2)

        return reg_loss

    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        normal_loss = np.mean(sample_losses)

        return normal_loss
    



class BinaryCrossEntropy_loss(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        outputs = len(dvalues[0])
        samples = len(dvalues)

        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = (
            -(y_true / dvalues_clipped - (1 - y_true) / (1 - dvalues_clipped)) / outputs
        )
        self.dinputs = self.dinputs / samples
