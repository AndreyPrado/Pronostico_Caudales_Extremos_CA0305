import matplotlib.pyplot as plt
import seaborn as sns
from BaseDatos import BaseDatos

class Grafico(BaseDatos):
    
    def __init__(self, url):
        ''' Inicializa una instancia de la clase heredando la dirección de BaseDatos
            
            Parámetros
            ----------
            url : str
                Ruta del archivo csv
            
            Retorna
            -------
        '''
        super().__init__(url)
        self.__grafico = None
        
    #Método String
    def __str__(self):
        ''' Da una breve descripción de la clase
        
            Parámetros
            ----------
            
            Retorna
            -------
        '''
        return "Gráficos para la visualización de datos"
    

    #Método para graficar un boxplot    
    def boxplot(self, col = None):
        ''' Genera un gráfico de cajas para columnas numéricas.
        
            Parámetros
            ----------
            col : str o list
                Columna(s) a graficar. Si es None, usa todas las numéricas.
            
            Retorna
            -------
            fig : matplotlib.figure
                Figura con el gráfico generado
        '''
        df = self._datos
        
        if col is None:
            col = df.select_dtypes(include=["number"]).columns.tolist()
            
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(data = df[col], ax=ax)
        ax.set_title("Gráfico de Cajas")
        plt.tight_layout()
        
        self.__grafico = fig
        
        return fig
    
    #Método para hacer un gráfico de líneas
    def linea(self, col = None):
        ''' Genera un gráfico de líneas para columnas numéricas.

            Parámetros
            ----------
            col : str o list
                Columna(s) a graficar. Si es None usa todas las numéricas.
            
            Retorna
            -------
            fig  : matplotlib.figure
                Figura con el gráfico generado.
        '''
        df = self._datos
        
        if col is None:
            col = df.select_dtypes(include=["number"]).columns.tolist()
            
        fig, ax = plt.subplots(figsize=(10,5))
        sns.lineplot(data=df[col], ax=ax)
        ax.set_title("Gráfico de Líneas")
        plt.tight_layout()
        
        self.__grafico = fig
        
        return fig
    
    #Método para hacer un heatmap
    def heatmap(self):
        ''' Genera un mapa de calor de correlaciones entre variables numéricas.
        
            Parámetros
            ----------
            
            Retorna
            -------
            
                fig : matplot.figure
                    Figura con el heatmap de correlaciones
        '''
        
        df = self._datos.select_dtypes(include=["number"])
        corr = df.corr()
       
        fig, ax = plt.subplots(figsize=(20,20))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Mapa de Calor de Correlaciones")
        plt.tight_layout()
       
        self.__grafico = fig
       
        return fig
   
    #Método para graficas las distribuciones
    def dist(self, col = None):
        ''' Grafica la distribución de una columna con un histograma
        
            Parámetros
            ----------
            
            col : str o list
                Nombre(s) de la(s) columna(s) a graficar.
            
            Retorna
            -------
            
            fig : matplotlib.figure
                Figura con el gráfico de distribución
        '''
        df = self._datos
        
        if col is None:
            col = df.select_dtypes(include=["number"]).columns.tolist()

        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df[col], kde = True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        plt.tight_layout()
            
        self.__grafico = fig
            
        return fig
        
    #Método para guardar los gráficos en png
    def guardar_en_png(self, nombre: str):
        ''' Guarda el gráfico actual en un archivo PNG.
    
            Parámetros
            ----------
            
            nombre : str
                Nombre del archivo (sin extensión) donde se guardará el gráfico.
        
            Retorna
            -------
            
            str
                Mensaje indicando el resultado de la operación.
        '''
        if self.__grafico is None:
            return "No se ha generado ningún gráfico"
        else:
            self.__grafico.savefig(nombre+".png")
        
        return f"Gráfico guardado como {nombre}.png"