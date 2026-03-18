# Combinational ML Power Estimation (NAND + NOR)

This project reproduces the combinational circuit (NAND and NOR) power estimation part of a research paper using Machine Learning.

The paper is here: [Intelligent Power Estimation of Digital Circuits Using Random Forest and Neural Network Models](https://link.springer.com/article/10.1007/s44196-025-01000-5)

## Overview

- Predicts power consumption of NAND and NOR CMOS circuits  
- Uses dataset provided in the paper  
- Focuses only on **combinational circuits** (no ISCAS’89 sequential circuits)  
- Implements and optimizes **Random Forest Regression**

## Features Used

- VL (Voltage Level)
- HL (Hierarchy Level)
- Number of Gates
- Inputs / Outputs
- Derived features:
  - Interaction terms
  - Gate density
  - Structural complexity

## Model

- Random Forest Regressor
- Hyperparameter tuning using Grid Search

## Results

| Circuit | Correlation | Deviation |
|--------|------------|-----------|
| NAND   | ~0.984     | ~19%      |
| NOR    | ~0.986     | ~19%      |

## Key Observations

- High correlation → model captures trend well  
- Higher deviation → dataset lacks physical parameters (capacitance, switching activity, etc.)  
- Exact reproduction of paper results (<1% deviation) is not achievable with given data  

## How to Run

```bash
python main.py