import nn
import numpy as np
from preprocessing import preprocess_dataset, preprocess_image


def get_batches(X, y, batch_size=100):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def train(training_set):
    np.random.seed(39)

    X, y = preprocess_dataset(training_set)

    dense1 = nn.Dense_layer(784, 128, weights_regularizer=1.2e-10)
    dense2 = nn.Dense_layer(128, 64, weights_regularizer=0)
    dense3 = nn.Dense_layer(64, 16, weights_regularizer=0)
    dense4 = nn.Dense_layer(16, 1, weights_regularizer=0)

    dropout1 = nn.Dropout(0.05)
    dropout2 = nn.Dropout(0.0)
    dropout3 = nn.Dropout(0.0)
    dropout4 = nn.Dropout(0.0)

    activation1 = nn.ReLU_activation()
    activation2 = nn.ReLU_activation()
    activation3 = nn.ReLU_activation()
    activation4 = nn.Sigmoid_activation()

    loss = nn.BinaryCrossEntropy_loss()
    optimizer = nn.Adam_optimzer(
        learning_rate=0.0001, decay=1e-6, beta_1=0.9, beta_2=0.999
    )

    for epoch in range(6):
        epoch_losses = []
        epoch_accs = []

        for X_batch, y_batch in get_batches(X, y):

            dense1.forward(X_batch)
            dropout1.forward(dense1.output)
            activation1.forward(dropout1.output)

            dense2.forward(activation1.output)
            dropout2.forward(dense2.output)
            activation2.forward(dropout2.output)

            dense3.forward(activation2.output)
            dropout3.forward(dense3.output)
            activation3.forward(dropout3.output)

            dense4.forward(activation3.output)
            dropout4.forward(dense4.output)
            activation4.forward(dropout4.output)

            normal_loss = loss.calculate(activation4.output, y_batch)
            reg_loss = (
                loss.regularization_loss(dense1)
                + loss.regularization_loss(dense2)
                + loss.regularization_loss(dense3)
            )
            total_loss = normal_loss + reg_loss

            predictions = (activation4.output > 0.5).astype(int)
            acc = np.mean(y_batch == predictions)

            epoch_losses.append(total_loss)
            epoch_accs.append(acc)

            loss.backward(activation4.output, y_batch)

            activation4.backward(loss.dinputs)
            dropout4.backward(activation4.dinputs)
            dense4.backward(dropout4.dinputs)

            activation3.backward(dense4.dinputs)
            dropout3.backward(activation3.dinputs)
            dense3.backward(dropout3.dinputs)

            activation2.backward(dense3.dinputs)
            dropout2.backward(activation2.dinputs)
            dense2.backward(dropout2.dinputs)

            activation1.backward(dense2.dinputs)
            dropout1.backward(activation1.dinputs)
            dense1.backward(dropout1.dinputs)

            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.update_params(dense3)
            optimizer.update_params(dense4)
            optimizer.post_update_params()

        print(f"Epoch: {epoch}, Acc: {np.mean(epoch_accs):.3f}, Loss: {np.mean(epoch_losses):.3f}")

    np.savez(
        "zero_or_one_validator_model.npz",
        w1=dense1.weights, b1=dense1.biases,
        w2=dense2.weights, b2=dense2.biases,
        w3=dense3.weights, b3=dense3.biases,
        w4=dense4.weights, b4=dense4.biases,
    )



    