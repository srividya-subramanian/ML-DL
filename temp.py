import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt 

Komponente = ["Grafikkarte", "Prozessor", "Festplatte", "Arbeitsspeicher", "Gehäuse"]
Dollar = [250, 300, 100, 80, 120]
Euro = [210, 252, 84, 67.2, 100.8]

xx = [Komponente, Dollar, Euro]

df1=pd.DataFrame(Komponente, columns=['Komponente'])
df2=pd.DataFrame(Dollar, columns=['Preis in Dollar'])
df3=pd.DataFrame(Euro, columns=['Preis in Euro'])
df = pd.concat([df1,df2,df3],axis=1)

exr1=(df['Preis in Dollar']/df['Preis in Euro']).mean()

def mse(actual,predicted):
    return np.mean((actual-predicted)**2)

result = sm.OLS(Dollar, Euro).fit()
exr2=result.params

plt.scatter(dollar_prices, euro_prices, color='blue', label='actual') 
plt.plot(dollar_prices, predicted_euros, color='red', label='predicted') 
plt.xlabel('Preis in Dollar') 
plt.ylabel('Preis in Euro') 
plt.legend() 
plt.show() 

