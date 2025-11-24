# Blood-Donation-Analysis

Using a dataset, train a Machine Learning model to predict whether a person will donate blood or not.

Here, the provided dataset is biased, using all accuracy development methods we came up with 83% accuracy as the most efficient one using xgboost.

## Frontend Application

This repository includes a complete frontend interface for the blood donation prediction model. The frontend is a simple, clean web application that allows users to input donor information and receive predictions.

## How to Run the Application

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python workbook.py
   ```

3. The application will automatically open in your default web browser at `http://localhost:5001`

   If port 5001 is already in use, you may need to kill the process using that port or modify the code to use a different port.

## Using the Application

1. Enter the donor information in the form:
   - Months since last donation
   - Number of donations
   - Total volume donated (c.c.)
   - Months since first donation

2. Click "Predict Donation" to get the prediction result

The model will analyze the data and predict whether the person will donate blood in March 2007, along with the confidence level of the prediction.

## API Usage

You can also use the API directly by sending a POST request to `http://localhost:5001/predict` with the following JSON payload:

```json
{
  "monthsLast": 10,
  "numDonations": 5,
  "totalVolume": 1250,
  "monthsFirst": 36
}
```

The response will be in the format:
```json
{
  "prediction": "Will Donate",
  "probability": 0.85
}
```