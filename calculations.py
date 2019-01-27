import pandas as pd
import numpy as np


def remove_outliers(data_frame):
    price = list(data_frame['P(t+1)'])
    q1 = np.percentile(price, 25)
    q3 = np.percentile(price, 75)

    in_range = [q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)]
    position = np.concatenate((np.where(price > in_range[1]),
        np.where(price < in_range[0])), axis=1)
    data_frame = data_frame.drop(data_frame.index[position[0]])

    return data_frame

def calculate_coefficient_matrix(data_frame):
  t2 = list(data_frame['T(t-2)'])
  t1 = list(data_frame['T(t-1)'])
  t = list(data_frame['T(t)'])
  d2 = list(data_frame['D(t-2)'])
  d1 = list(data_frame['D(t-1)'])
  d = list(data_frame['D(t)'])
  p = list(data_frame['P(t+1)'])

  A = np.row_stack((t2, t2, t, d2, d2, d, p))

  return np.corrcoef(A)

training_data_frame = pd.read_csv("datasets/Training_Data.csv")
testing_data_frame = pd.read_csv("datasets/Testing_Data.csv")

training_data_frame = remove_outliers(training_data_frame)
testing_data_frame = remove_outliers(testing_data_frame)

training_CCM = calculate_coefficient_matrix(training_data_frame)
testing_CCM = calculate_coefficient_matrix(testing_data_frame)

print(training_CCM)
print(testing_CCM)
