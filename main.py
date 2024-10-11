import numpy as np
from thermo.group_contribution import Joback
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import joblib

def smiles_to_model_input(smiles: str) -> np.ndarray:
    '''
    Converts a SMILES string into an input feature array suitable for the MLP model.
    
    This function takes the SMILES (Simplified Molecular Input Line Entry System) representation of a molecule 
    and converts it into a feature vector. The first element of the vector is the exact molecular weight of 
    the compound, and the remaining elements correspond to the Joback group contributions.

    Parameters:
    ----------
    smiles : str
        The SMILES string representing the molecular structure of a compound.

    Returns:
    -------
    np.ndarray
        A NumPy array of shape (42,), where:
        - The first element contains the exact molecular weight.
        - The remaining 41 elements represent the Joback subgroups present in the molecule. 
          These are initialized to zero if no such group is present in the molecule.
    
    Raises:
    ------
    ValueError:
        If the SMILES string is invalid and cannot be parsed by RDKit.

    Example:
    -------
    >>> smiles_to_model_input("CCO")
    array([46.069, 1.0, 0.0, 0.0, ..., 0.0])
    '''
    
    sub_groups_dict = Joback(smiles).counts
    input_array = np.zeros(42)
    input_array[0] = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
    for i in range(1, 42):
        input_array[i] = sub_groups_dict.get(i, 0)
    return input_array

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(42,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def predict_flash_point(model, scaler, smiles: str) -> float:
    '''
    Predicts the flash point temperature of a given chemical compound based on its SMILES string.

    This function takes the SMILES string of a molecule, converts it into the model's input feature format, 
    scales the input using the provided scaler, and then uses the trained model to predict the flash point temperature.

    Parameters:
    ----------
    model : tensorflow.keras.Model
        The trained TensorFlow model used for predicting the flash point temperature.
    
    scaler : sklearn.preprocessing.MinMaxScaler or similar
        The scaler used to normalize the input features. It should be the same scaler that was used to normalize 
        the training data.
    
    smiles : str
        The SMILES string representing the chemical compound whose flash point temperature is to be predicted.

    Returns:
    -------
    float
        The predicted flash point temperature of the chemical compound.

    Raises:
    ------
    ValueError:
        If the SMILES string is invalid and cannot be processed or if there is an issue with the input data.

    Example:
    -------
    >>> predict_flash_point(model, scaler, "CCO")
    307.5
    '''

    input_array = smiles_to_model_input(smiles).reshape(1, -1)
    input_array = scaler.transform(input_array)
    return model(input_array)

if __name__ == '__main__':
    model = create_model()
    model.load_weights('models/FPT_Joback.h5')

    scaler = scaler = joblib.load('models/FPT_Joback.pkl')

    # take the smiles from user
    smiles = 'CCCC=CCc1ccccc1'

    try:
        fpt = predict_flash_point(model, scaler, smiles)
        print('-'*100)
        print(f'Estimated Flash Point Temperature for {smiles}: \n{float(fpt[0, 0]): .3f}')
        print('-'*100)
    except ValueError as e:
        raise ValueError(f"An error occurred while estimating flash point temperature: {e}")


