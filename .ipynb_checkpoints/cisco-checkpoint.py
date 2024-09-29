import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Cargar y preparar los datos
data = pd.read_csv('/Users/claudiogonzalezarriaga/Documents/Progra_Tec/QuintoSemestre/metodos multivariados/cisco forecast/cisco_data.csv')
data = data.melt(id_vars=['Cost Rank', 'Product Name', 'PLC'], 
                 var_name='Date', value_name='Value')
data['Date'] = pd.to_datetime(data['Date'].str.replace('FY', '20') + '-01-01')
data['Value'] = pd.to_numeric(data['Value'].str.replace(',', ''), errors='coerce')

# Función para entrenar y predecir con Prophet
def train_predict_prophet(df):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=1, freq='Q')
    forecast = model.predict(future)
    return model, forecast

# Función para calcular MAPE
def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

results = []

for product in data['Product Name'].unique():
    df = data[data['Product Name'] == product].copy()
    df = df[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})
    df = df.sort_values('ds')
    
    if len(df) > 4:  # Asegurarse de que hay suficientes datos para el entrenamiento
        model, forecast = train_predict_prophet(df)
        
        # Validación cruzada
        cv_results = cross_validation(model, initial='730 days', period='90 days', horizon='90 days')
        cv_metrics = performance_metrics(cv_results)
        mape = cv_metrics['mape'].mean()
        
        # Predicción para FY2024 Q4
        prediction = forecast.iloc[-1]['yhat']
        
        results.append({
            'Product Name': product,
            'FY2024 Q4 Prediction': prediction,
            'MAPE': mape
        })

# Crear DataFrame con resultados
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('MAPE')

print(results_df)

# Calcular MAPE promedio
average_mape = results_df['MAPE'].mean()
print(f"\nMAPE promedio del modelo: {average_mape:.2%}")