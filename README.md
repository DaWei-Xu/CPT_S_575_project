# CPT_S_575 Project

A course project for **CPT_S 575 – Data Science** at **Washington State University**.  
This repository contains Python scripts, Jupyter notebooks, and datasets used for training, evaluating, and visualizing neural network models.

The main goal of this project is to apply machine learning techniques to waste water data analysis. 

The repository includes complete workflows for:

- Data preprocessing and encoding  
- Neural network model setup
- Training and validation of models  
- Loss curve visualization and learning rate tracking  
- Comparison of different architectures and hyperparameters


## Environment Setup

You can set up the environment using **Conda**.

### Clone the repository

```bash
git clone https://github.com/DaWei-Xu/CPT_S_575_project.git
cd CPT_S_575_project
```

### Create and activate the environment

```bash
conda create -n env_cpts575 python=3.12
conda activate env_cpts575
```

### Install required packages

```bash
pip install -r requirements.txt
```

## Usage
### Process the raw dataset and prepare encoders:

```bash
python src/data_preprocess.py
```

### Training and visualizations

```bash
python training_scripts/casexxx.py
```

> Replace `casexxx` with the specific case number (e.g., `case000`).

The trained models and corresponding figures can be found in directory:

```
data/training_results/casexxx/
```

## Authors

**Dawei Xu**  
Washington State University  
[dawei.xu@wsu.edu](mailto:dawei.xu@wsu.edu)

**Yuqun Song**  
Washington State University  
[yuqun.song@wsu.edu](mailto:yuqun.song@wsu.edu)

## License

This project is for **academic and research purposes** only.  
Users are welcome to reference or adapt the code with proper attribution.

