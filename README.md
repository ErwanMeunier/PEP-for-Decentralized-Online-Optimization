# Distributed Online Optimization Algorithms

This repository provides MATLAB and Julia implementations of several **decentralized online optimization (DOO) algorithms**, including Distributed Autonomous Online Learning (DAOL), Distributed Online Conditional Gradient (DOCG), and Distributed Online Mirror Descent (DOMD).  

Beyond implementation, the repository leverages the **Performance Estimation Problem (PEP) framework** to compute **tight worst-case performance bounds** for these methods. This allows practitioners and researchers to (i) benchmark algorithms more accurately, (ii) avoid overly conservative analytical bounds that can mislead method selection, and (iii) improve algorithms through step-size tuning and design, achieving significantly better worst-case regret guarantees.  

The code is meant to accompany our forthcoming paper, serving as a practical resource for reproducing experiments and exploring method improvements.  

---

## Quick Refresher

In **distributed online optimization**, multiple agents cooperate to make sequential decisions while only observing information revealed over time.  
Each agent updates its local model using limited feedback, communicates with neighbors, and together they aim to minimize global regret.  
The methods implemented here explore different algorithmic strategies (learning rates, mirror descent, conditional gradients) to achieve efficiency in this setting.  

---

## Implemented Methods

- **DAOL – Distributed Autonomous Online Learning**  
  MATLAB implementations of distributed online learning with different step-size schemes.

- **DOMD – Distributed Mirror Descent for Online Composite Optimization**  
  MATLAB and Julia implementations, including performance estimation problems (PEP) and primal bound simulations.

- **DOCG – Distributed Online Conditional Gradient**  
  MATLAB implementation of conditional gradient methods in the distributed online setting.

---

## Repository Structure

```
LICENSE
Distributed Autonomous Online Learning/
  ├─ bound_daol.m
  ├─ distributed_autonomous_online_learning.m
  └─ distributed_autonomous_online_learning_given_step_sizes.m

Distributed Mirror Descent for Online Composite Optimization/
  ├─ PEP/
  │   ├─ bound_DOMD.m
  │   └─ distributed_mirror_descent_online_optimization.m
  └─ Primal bound for the worst-case function of DOMD/
      ├─ generating_doubly_stochastic_matrices.jl
      └─ Simulating_DOMD_withoutPEP.jl

Distributed Online Conditional Gradient/
  ├─ bound_docg.m
  └─ distributed_online_conditional_gradient.m

Method design/
  └─ design_DAOL.m
```

---

## Requirements

- **MATLAB** (R2020a or later recommended)  
- **Julia** (v1.7 or later)  
  - Required Julia packages:
    - `LinearAlgebra`
    - `Random`
    - (others will be listed in the corresponding `.jl` files)

---

## Usage

### MATLAB examples
Run any of the provided `.m` scripts in MATLAB. For instance:
```matlab
>> distributed_autonomous_online_learning
```

### Julia examples
From the `Primal bound for the worst-case function of DOMD` folder, run:
```bash
julia Simulating_DOMD_withoutPEP.jl
```

---

## License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.

---

## Citation

If you use this code in your research, please cite the accompanying paper:

```
@article{your_paper_2025,
  title   = {Title of Your Paper},
  author  = {Author Names},
  journal = {Journal/Conference},
  year    = {2025},
}
```
