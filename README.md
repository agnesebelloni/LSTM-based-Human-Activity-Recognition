# LSTM-based-Human-Activity-Recognition
LSTM-based Human Activity Recognition using wearable sensor time-series data. 

# Human Activity Recognition with LSTM

This project implements a Human Activity Recognition system using an LSTM neural network on time-series sensor data.

The goal is to classify human activities from sequential accelerometer and gyroscope signals.

## Activities

The model classifies activities such as:

- Walking
- Walking upstairs
- Walking downstairs
- Sitting
- Standing
- Laying

## Method

The input signals are preprocessed and organized as time windows.  
An LSTM network is trained to learn temporal patterns from the sensor sequences and classify each window into the correct activity class.

## Technologies

- Python
- NumPy
- Pandas
- PyTorch / TensorFlow
- Scikit-learn
- Matplotlib

## Results

The model was evaluated using accuracy and confusion matrix.

## Run

```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
