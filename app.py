from flask import Flask, request, render_template, send_file
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import io

model = load_model('model.h5')
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

def make_prediction(time_input):
    df = pd.read_csv('dataset.csv')
    df['Login Hour'] = pd.to_datetime(df['Login Time']).dt.hour
    df['Logout Hour'] = pd.to_datetime(df['Logout Time']).dt.hour

    filtered = df[(df['Login Hour'] <= time_input) & (df['Logout Hour'] >= time_input)]

    if filtered.empty:
        return pd.DataFrame()

    filtered_encoded = encoder.transform(filtered[['State', 'Region', 'Speciality']])
    filtered_input = np.concatenate(
        [
            filtered_encoded,
            filtered[['Login Hour', 'Logout Hour', 'Usage Time (mins)', 'Count of Survey Attempts']].values
        ],
        axis=1
    )

    filtered_input_scaled = scaler.transform(filtered_input)
    filtered['Probability'] = model.predict(filtered_input_scaled).flatten()
    print(type(filtered['Probability']))
    filtered['Probability'] = filtered['Probability'].apply(lambda x: str(x*100)[:6])


    best_doctors = filtered.sort_values(by='Probability', ascending=False).head(10)

    return best_doctors

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        action = request.form.get('action')
        time_input = int(request.form['time'])

        best_doctors = make_prediction(time_input)

        if action == 'predict':
            if best_doctors.empty:
                message = 'No Doctor Found at This Time!!'
                return render_template('index.html', best_doctors=None, message=message)

            return render_template('index.html', best_doctors=best_doctors, time_input=time_input, message=None)

        elif action == 'download_csv':
            if not best_doctors.empty:
                csv_buffer = io.StringIO()
                best_doctors.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                return send_file(
                    io.BytesIO(csv_buffer.getvalue().encode()),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='best_doctors.csv'
                )

        elif action == 'download_excel':
            if not best_doctors.empty:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    best_doctors.to_excel(writer, index=False)
                excel_buffer.seek(0)
                return send_file(
                    excel_buffer,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name='best_doctors.xlsx'
                )

    return render_template('index.html', message=None, time_input=None, best_doctors=None)

if __name__ == '__main__':
    app.run(debug=True)
