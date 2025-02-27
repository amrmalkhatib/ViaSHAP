# **ViaSHAP - Prediction via Shapley Value Regression**

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## **Description:**
This repository hosts the implementation of ViaSHAP, a novel approach that learns a function to compute Shapley values, from which the predictions can be derived directly by summation. We explore two learning approaches based on the universal approximation theorem and the Kolmogorov-Arnold representation theorem. ViaSHAP using Kolmogorov-Arnold Networks performs on par with state-of-the-art algorithms for tabular data. The explanations obtained using ViaSHAP are significantly more accurate than other popular approximators, e.g., FastSHAP on both tabular data and images. All the experiments have been conducted in a Python 3.8 environment.
## **Usage:**
1. Clone the Repository: Clone this repository to your local machine using the following command:
   ```
   git clone https://github.com/amrmalkhatib/ViaSHAP.git
   ```
2. Install Dependencies: Ensure that you have the necessary dependencies installed. You can install them using `pip`:
   ```
   pip install -r requirements.txt
   ```
3. Sample notebooks are provided