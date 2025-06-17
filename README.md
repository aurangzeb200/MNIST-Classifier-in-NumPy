# Deep Learning from Scratch: MNIST Classifier in NumPy

A complete neural network built from scratch using only NumPy to classify handwritten digits from the MNIST dataset. No TensorFlow, no PyTorch — just matrix math, clean logic, and gradient descent.

This project covers full implementation of a feedforward neural network including activation functions, forward/backward propagation, weight updates, and loss calculations. Multiple experiments are included for comparing learning rates, activation functions, architectures, and loss functions.

## 🧠 Techniques of Interest

- Manual implementation of [softmax](https://en.wikipedia.org/wiki/Softmax_function) for multi-class output
- Vectorized [matrix operations](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) with NumPy for fast computation
- Custom [activation functions](https://en.wikipedia.org/wiki/Activation_function): `sigmoid`, `tanh`, and `softmax`
- Manual [mini-batch gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Mini-batch_gradient_descent)
- Comparison of [MSE vs. Cross-Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy) for classification

## 🔧 Libraries and Tools Used

- [NumPy](https://numpy.org/) – numerical operations and vector math
- [Matplotlib](https://matplotlib.org/) – image visualization of sample digits
- [Scikit-learn](https://scikit-learn.org/stable/) – only used to load the MNIST dataset via `fetch_openml`
- [Google Colab](https://colab.research.google.com/) – for running the `.ipynb` notebook interactively
- [OpenML](https://www.openml.org/d/554) – MNIST dataset source

> This project does not use any front-end, UI libraries, or custom fonts.

## 📁 Project Structure

- **`mnist_numpy_nn.ipynb`** – Implements data loading, preprocessing, training, forward/backward propagation, and experiments.
- **`/images/`** – Use this to store training visualizations or example outputs.
- **`/notebooks/`** – Useful for extensions, alternate models, or comparative studies.

## 📌 Note

This project is designed for educational purposes and low-level understanding of how neural networks work internally. It avoids any deep learning abstraction layers and is ideal for developers looking to demystify backpropagation and activation logic.

---
