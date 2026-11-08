import pandas as pd
from scipy import stats
def group_stats(df: pd.DataFrame) -> dict:
    """Genera tablas de agregación (Capa Gold) agrupando por diferentes variables."""
    
    # 1. Edad por supervivencia (usando describe)
    age_by_survival = df.groupby('Survived')['Age'].describe().reset_index()
    
    # 2. Supervivencia por Sexo
    survival_by_sex = df.groupby('Sex')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    
    # 3. Supervivencia por Clase
    survival_by_pclass = df.groupby('Pclass')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    
    # 4. Supervivencia por 'Alone' (Solo vs Familia)
    survival_by_alone = df.groupby('Alone')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    
    # Devolvemos todas en un diccionario
    return {
        'age_by_survival': age_by_survival,
        'survival_by_sex': survival_by_sex,
        'survival_by_pclass': survival_by_pclass,
        'survival_by_alone': survival_by_alone
    }

def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Devuelve {'chi2': float, 'p_value': float, 'dof': int}."""
    
    # 1. Creamos la tabla de contingencia (frecuencias observadas)
    tabla = pd.crosstab(df['Sex'], df['Survived'])
    
    # 2. Aplicamos la prueba de Chi-cuadrado
    chi2, p, gl, esperados = stats.chi2_contingency(tabla)
    
    return {
        'chi2': chi2,
        'p_value': p,
        'dof': gl
    }