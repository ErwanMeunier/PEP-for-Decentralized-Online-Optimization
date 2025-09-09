# Distributed Online Optimization Algorithms

Beyond implementation, the repository leverages the **Performance Estimation Problem (PEP) framework** to compute **tight worst-case performance bounds** of several **decentralized online optimization (DOO) algorithms**, including Distributed Autonomous Online Learning (DAOL), Distributed Online Conditional Gradient (DOCG), and Distributed Online Mirror Descent (DOMD). This allows practitioners and researchers to (i) benchmark algorithms more accurately, (ii) avoid overly conservative analytical bounds that can mislead method selection, and (iii) improve algorithms through step-size tuning and design, achieving significantly better worst-case regret guarantees.  

The code is meant to accompany our forthcoming paper, serving as a practical resource for reproducing experiments and exploring method improvements.  

---

## Quick Refresher

In **distributed online optimization**, multiple agents cooperate to make sequential decisions while only observing information revealed over time.  
Each agent updates its local model using limited feedback, communicates with neighbors, and together they aim to minimize global regret.  
The methods implemented here explore different algorithmic strategies (gradient descent, mirror descent, conditional gradients) to achieve efficiency in this setting.  

---

## Implemented Methods

- **DAOL – Distributed Autonomous Online Learning**  

- **DOMD – Distributed Mirror Descent for Online Composite Optimization**  

- **DOCG – Distributed Online Conditional Gradient**  

---

## Requirements

- **MATLAB** (R2020a or later recommended)  
- **PESTO TOOLBOX** --> https://github.com/PerformanceEstimation/Performance-Estimation-Toolbox/
  Some **PESTO files must be changed** to include the support of Strongly Convex Functions with Bounded Gradients (1) not directly available in the PESTO toolbox.
  You can use the `PESTO_fork_standalone.zip` which is a fork of the PESTO toolbox including the support of (1).
---

## Usage

### MATLAB examples
The `dedicated_simulations.m` provides a way of directly analyzing DOO algorithms by using PESTO.

## License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.

---

## Citation

If you use this code in your research, please cite the accompanying paper:

```
@misc{meunier2025performanceboundsdecentralizedonline,
      title={Several Performance Bounds on Decentralized Online Optimization are Highly Conservative and Potentially Misleading}, 
      author={Erwan Meunier and Julien M. Hendrickx},
      year={2025},
      eprint={2509.06466},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2509.06466}, 
}
```
