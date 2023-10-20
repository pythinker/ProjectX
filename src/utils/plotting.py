import matplotlib.pyplot as plt


def plot_predictions(train_dataset, test_dataset, y_pred_train, y_pred_test):

    X_train = train_dataset.X
    y_train = train_dataset.y

    X_test = test_dataset.X
    y_test = test_dataset.y

    y_pred_train = y_pred_train.detach().numpy()
    y_pred_test = y_pred_test.detach().numpy()

    y_pred_train = y_pred_train > 0.5
    y_pred_test = y_pred_test > 0.5

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    axs[0, 0].set_title('Train Ground Truth Labels')
    axs[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train)
    axs[0, 1].set_title('Train Predicted Labels')
    axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    axs[1, 0].set_title('Test Ground Truth Labels')
    axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test)
    axs[1, 1].set_title('Test Predicted Labels')
    fig.tight_layout()

    return fig
