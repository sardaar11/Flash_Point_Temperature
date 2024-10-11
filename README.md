# Flash Point Temperature Prediction using MLP

This project predicts the flash point temperature of chemical compounds using Joback subgroups as input features. A multi-layer perceptron (MLP) model is trained using TensorFlow for the task. The model maps the number of subgroups in a molecule to its corresponding flash point temperature.

## Model Details

- **Input features**: Joback subgroups (number of subgroups for each compound)
- **Labels**: Flash Point Temperature (K)
- **Framework**: TensorFlow
- **Evaluation Metrics**: MAE, MSE, R²

## Repository Contents

- `Flash_Point_Temperature.ipynb`: Jupyter notebook containing the model training and evaluation.
- `model/`: Directory containing saved models.
- `data/`: Sample dataset with chemical compound SMILES and flash point temperatures.
- `requirements.txt`: Dependencies for running the project.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flash-point-prediction.git
2. Run the main.py.
3. write the SMILES of the chemical compound to predict its Flash Point Temperature.
