# The Fluid Dynamic Neural Network

## Overview
The **Fluid Dynamic Neural Network (FDNN)** is a groundbreaking deep learning architecture inspired by **fluid dynamics**. Instead of relying on traditional weight-based neural networks, FDNN models information processing as **fluid flow**, where activations evolve over time based on diffusion, viscosity, and wave propagation principles.

This repository provides an implementation of FDNN, including:
- A fully differentiable **FluidLayer**, simulating wave-based feature propagation.
- A multi-layer **FullyFluidNetwork** designed for image classification.
- Training and evaluation on the **MNIST handwritten digit dataset**.

FDNN is released under the **MIT License**, making it open and accessible for research, experimentation, and real-world applications.

---

## Features
✅ **Fluid-Based Computation:** Layers behave as **fluid fields**, where activations evolve dynamically over multiple time steps.  
✅ **Entropy-Driven Learning:** Uses diffusion, viscosity, and wave interference instead of standard matrix multiplications.  
✅ **Improved Stability & Generalization:** Simulates smooth information flow, potentially leading to better learning efficiency.  
✅ **No Backpropagation Needed (Future Goal):** FDNN aims to develop into a truly self-organizing, non-gradient-based AI system.  
✅ **Plug-and-Play:** Works with PyTorch, making it easy to integrate into existing deep learning workflows.  

---

## Installation
### **Prerequisites**
- Python 3.8+
- PyTorch
- Torchvision
- NumPy

### **Setup**
Clone the repository and install dependencies:
```bash
$ git clone https://github.com/yourusername/FluidDynamicNN.git
$ cd FluidDynamicNN
$ pip install -r requirements.txt
```

---

## Usage
### **Training the Model**
To train the FDNN on MNIST:
```bash
$ python train.py
```

### **Testing the Model**
To evaluate on the test set:
```bash
$ python test.py
```

---

## Architecture
The core of FDNN is the **FullyFluidLayer**, which replaces traditional dense layers with an **entropy-controlled wave propagation model**. Each layer consists of:
- **Diffusion Control:** Allows feature propagation across the layer.
- **Viscosity Parameter:** Stabilizes chaotic feature transitions.
- **Velocity Fields:** Simulates information movement over time.

A complete **FullyFluidNetwork** is constructed using stacked **FluidLayers**, with final classification performed by a readout layer.

---

## Example Code
Here’s a simple example of how to use FDNN in a PyTorch project:
```python
import torch
from model import FullyFluidNetwork

# Initialize model
model = FullyFluidNetwork(input_size=28*28, hidden_size=256, output_size=10, time_steps=50)

# Random input tensor
input_data = torch.randn(1, 28*28)
output = model(input_data)
print(output)
```

---

## Roadmap
🔹 Expand FDNN to support **larger datasets** (e.g., CIFAR-10, ImageNet).  
🔹 Explore **self-organizing fluid dynamics** as a replacement for backpropagation.  
🔹 Optimize for **real-time applications** in robotics and reinforcement learning.  
🔹 Implement **hardware acceleration** for fluid-based AI on neuromorphic chips.  

---

## License
This project is licensed under the **MIT License**—free to use, modify, and distribute.

---

## Acknowledgments
This project was inspired by **fluid physics, entropy-based learning, and non-standard AI paradigms**. Contributions and collaborations are welcome!

For inquiries, reach out via GitHub issues or email.

🚀 *Let's push the boundaries of AI together!* 🌊
