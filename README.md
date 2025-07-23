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
- --steps, -s: Maximum number of training iterations  
- --init_theta0, --init_theta1: Initial values for the model parameters
- --plot, -p: Displays graphs of the training process and regression

### 📁 Sample data.csv file  

```csv
km,price
240000,3650
139800,3800
150500,4400
...
``` 


## 🔍 Prediction  

Once the model is trained, predict the price of a car from the command line:  

```bash
python predict.py --jsonfile
```
Or by manually specifying the parameters:  

```bash
python predict.py -th0 <theta0_value> -th1 <theta1_value>
```

## 📊 Visualization  

If you use the --plot flag during training, the following graphs will be displayed:
  + The regression line fitted to the dataset.
  + The Mean Squared Error (MSE) over training steps.  
  + The coefficient of determination (R²) showing model accuracy.

## ⚙️ Setting up the Environment  

This project uses Pipenv for dependency management.  
To install dependencies and create the virtual environment:  

```bash
chmod +x install.sh
./install.sh
```

Then, activate the environment with:  

```bash
pipenv shell
```

## ✅ Requirements  

+ Python 3.13
+ Pipenv
+ Libraries:
    + numpy
    + matplotlib  


