import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    input_data = [(x) for x in request.form.values()]
    if input_data[1] == "male":
        input_data[1] = 0.0
    else:
        input_data[1] = 1.0
    if input_data[4] == "yes":
        input_data[4] = 0.0
    else:
        input_data[4] = 1.0
    if input_data[5] == "southeast":
        input_data[5] = 0.0
    elif input_data[5] == "southwest":
        input_data[5] = 1.0
    elif input_data[5] == "northeast":
        input_data[5] = 2.0
    elif input_data[5] == "northwest":
        input_data[5] = 3.0
    input_d = float(input_data[2])
    # input_datanew=int(input_d)
    input_data[2] = input_d
    input_d = float(input_data[0])
    # input_datanew = int(input_d)
    input_data[0] = input_d
    input_d = float(input_data[3])
    # input_datanew = int(input_d)
    input_data[3] = input_d

    # encoding sex column
    # input_data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)

    # encoding 'smoker' column
    # input_data.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)

    # encoding 'region' column
    # input_data.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}},
    # inplace=True)

    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    # print('The insurance cost is USD ', prediction[0])
    return render_template("index.html", prediction_text="The insurance cost is Rs.{}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)