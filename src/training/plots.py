import matplotlib.pyplot as plt


def plot_loss(history):
    dice = history.history["dice"]
    val_dice = history.history["val_dice"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dice, label="Training Dice")
    plt.plot(val_dice, label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
