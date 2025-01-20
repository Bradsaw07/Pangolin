import pandas as pd
import sys
import os
from pathlib import Path

# Obtener la ruta absoluta del directorio actual
current_dir = Path(__file__).parent.absolute()

# Agregar el directorio src al path
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # Importar solo las funciones de EDA que vamos a usar
    from datautilityhub.eda import (
        generate_summary_statistics,
        plot_correlation_matrix,
        plot_distribution
    )
except ImportError as e:
    print(f"Error al importar: {e}")
    print(f"Python Path actual: {sys.path}")
    print(f"Directorio actual: {current_dir}")
    print(f"Buscando en: {src_path}")
    
    # Intentar importar directamente del archivo eda.py
    try:
        import sys
        sys.path.append(src_path)
        from datautilityhub.eda import (
            generate_summary_statistics,
            plot_correlation_matrix,
            plot_distribution
        )
    except Exception as e:
        print(f"Segundo intento fallido: {e}")
        sys.exit(1)

def main():
    # Crear un DataFrame de ejemplo
    data = {
        'edad': [25, 30, 35, 40, 45, 28, 32, 38, 42, 47],
        'salario': [30000, 45000, 50000, 60000, 70000, 35000, 48000, 55000, 65000, 75000],
        'experiencia': [1, 5, 7, 10, 15, 2, 6, 8, 12, 17],
        'satisfaccion': [7, 8, 6, 8, 9, 7, 8, 7, 8, 9]
    }
    
    df = pd.DataFrame(data)
    
    # Generar estadísticas resumidas
    print("\n=== Estadísticas Resumidas ===")
    summary_stats = generate_summary_statistics(df)
    print(summary_stats)
    
    # Crear visualizaciones
    print("\n=== Generando Visualizaciones ===")
    
    # Matriz de correlación
    plot_correlation_matrix(df, 
                          title="Matriz de Correlación - Datos de Empleados")
    
    # Distribuciones de variables numéricas
    for columna in df.columns:
        plot_distribution(df[columna], 
                        title=f"Distribución de {columna}")
    
    print("\nAnálisis completado. Revisa las visualizaciones generadas.")

if __name__ == "__main__":
    main()
