# 🧠 ft_linear_regression

Project completed as part of 42 School, introducing the basics of machine learning through simple linear regression using gradient descent.

## 📈 Objective

Predict the price of a car based on its mileage using a linear regression model trained with the gradient descent algorithm.

---

## 🛠️ Project Structure

- `train.py`: Trains the model using gradient descent and saves the `theta0` and `theta1` parameters to a JSON file.
- `linear_regression.py`: Core module containing the training logic, visualization, and model evaluation metrics.
- `predict.py`: Loads the trained parameters and predicts the price of a car given its mileage.
- `install.sh`: Script to install `pipenv` and set up the virtual environment.
- `Pipfile`: Lists required Python dependencies.

---

## 🧪 Training the Model

To train the model, run:

```bash
python train.py data.csv --plot
```

### Useful Arguments:
- --delimiter, -d: CSV file delimiter (default is ,)
- --learning_rate, -lr: Learning rate for gradient descent

