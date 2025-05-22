import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from Modelo import Modelo as Modelo

class ModeloRandomForest(Modelo):
    def __init__(self, url, target_column, col_ignore):
        super().__init__(url)
        self.__target_column = target_column
        self.__col_ignore = col_ignore
        self.__model = RandomForestRegressor(n_estimators=100, random_state=42)



    #Getters y Setters 

    
    def ajustar_modelo(self):

        self._X_train, self._X_test, self._Y_train, self._Y_test = self.cargar_datos(
            x=[self.__col_ignore], y=[self.__target_column])
        self.__model.fit(self._X_train, self._Y_train)

    def evaluar_modelo(self):
        # Puntaje R2
        r2_train = self.__model.score(self._X_train, self._Y_train)
        r2_test = self.__model.score(self._X_test, self._Y_test)
    
        print(f"R² (train): {r2_train:.4f}")
        print(f"R² (test): {r2_test:.4f}")
    
        # Validación cruzada
        scores = cross_val_score(self.__model, self._X_train, self._Y_train, cv=5, scoring='r2')
        print(f"R² promedio (CV): {scores.mean():.4f}")
    
        # Predicciones
        self.__y_pred = self.__model.predict(self._X_test)
    
        # Métricas de error
        mae = mean_absolute_error(self._Y_test, self.__y_pred)
        mse = mean_squared_error(self._Y_test, self.__y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(self._Y_test, self.__y_pred) * 100  # porcentaje
    
        # NSE - Nash-Sutcliffe Efficiency
        y_obs = self._Y_test.values.ravel()
        y_pred = self.__y_pred.ravel()
        nse = 1 - np.sum((y_obs - y_pred) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)
    
        print(f"""
        Métricas de Error:
        - MAE (Error Absoluto Medio): {mae:.2f}
        - MSE (Error Cuadrático Medio): {mse:.2f}
        - RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}
        - MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%
        - NSE (Eficiencia Nash–Sutcliffe): {nse:.4f}
        """)

    def visualizar_resultados(self):

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self._Y_test.values.ravel(), y=self.__y_pred.ravel(), alpha = 0.6)
        plt.plot([self._Y_test.min(), self._Y_test.max()], [self._Y_test.min(), self._Y_test.max()], 'r--')  # Línea de perfecta predicción
        plt.title('Valores Reales vs Predicciones')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.grid(True)
        plt.show()
        
    def importancia_feature(self, top_n = 10):
        importances = self.__model.feature_importances_
        feature_names = self._X_train.columns
        
        df_impor = pd.DataFrame({
            'feature' : feature_names,
            'importancia' : importances
            }).sort_values(by='importancia', ascending = False)
        
        print(df_impor.head(top_n))
        plt.figure(figsize=(10,6))
        sns.barplot( x = 'importancia', y='feature', data = df_impor.head(top_n), palette = 'viridis')
        plt.title('Importancia de las Variables del Modelo Random Forest')
        plt.tight_layout()
        plt.show()
        
    def importancia_permutacion(self, top_n = 10):
        feature_names = self._X_train.columns
        
        resultados = permutation_importance(self.__model,self._X_test, self._Y_test, n_repeats=10,random_state=42)
        df_permu = pd.DataFrame({
            'feature' : feature_names,
            'media_importancia' : resultados.importances_mean,
            'sd_importancia' : resultados.importances_std 
            }).sort_values(by = 'media_importancia', ascending=False)
        
        print(df_permu.head(top_n))
        
        plt.figure(figsize=(10,6))
        sns.barplot(x='media_importancia', y='feature', data = df_permu.head(top_n), palette = 'viridis')
        plt.title('Importancia de las variables del Modelo Random Forest por Permutación')
        plt.tight_layout()
        plt.show()
        
        
        