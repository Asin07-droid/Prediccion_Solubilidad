# Predicción de solubilidad acuosa mediante modelos QSAR

Proyecto de química computacional aplicada para estudiar la predicción de la solubilidad acuosa (`logS`) de compuestos orgánicos a partir de descriptores moleculares 2D. El flujo compara modelos de aprendizaje automático y examina la relación entre el número de descriptores, el rendimiento predictivo y el costo computacional.

## Objetivos

- Construir un flujo de trabajo reproducible para preparar datos, seleccionar descriptores, entrenar modelos y evaluar predicciones.
- Comparar Ridge, Random Forest, XGBoost y LightGBM.
- Estudiar el rendimiento en función del número de descriptores.
- Comparar una configuración de consenso con otra de mayor número de descriptores.
- Explorar alternativas que puedan ejecutarse en equipos de recursos modestos, sin asumir una ventaja computacional.

El proyecto es exploratorio y no pretende demostrar superioridad universal frente a otros métodos ni sustituir mediciones experimentales.

## Datos

El repositorio organiza los conjuntos originales y procesados en `data/raw/` y `data/processed/`.

- **AqSolDB:** colección de datos de solubilidad acuosa compilada a partir de distintas fuentes.
- **Delaney/ESOL:** conjunto de referencia para predicción de solubilidad acuosa.
- **SC-2 tight y SC-2 loose:** conjuntos externos de menor tamaño, utilizados para examinar el comportamiento fuera del conjunto principal.

La procedencia y las condiciones de redistribución de cada conjunto deben revisarse por separado. La licencia del código no implica automáticamente una licencia equivalente para los datos.

## Estructura y orden de ejecución

| Notebook | Propósito |
|---|---|
| `01_preprocesamiento.ipynb` | Preprocesamiento y preparación de descriptores moleculares. |
| `02_configuracion_y_datos.ipynb` | Configuración compartida y preparación de datos. |
| `03_seleccion_features_v4_eficiente_5folds.ipynb` | Selección de descriptores y barrido consecutivo de valores de `k` con validación cruzada. |
| `04_modelos_base.ipynb` | Comparación de modelos base. |
| `05_validacion_interna.ipynb` | Evaluación diagnóstica interna. |
| `06_optimizacion_dos_modelos.ipynb` | Optimización de modelos mediante Optuna. |
| `07_validacion_externa.ipynb` | Evaluación reportada sobre conjuntos externos. |

Ejecuta los notebooks en orden y desde el directorio `notebooks/`, respetando las rutas relativas definidas en el código.

## Barrido consecutivo de descriptores

El notebook 03 utiliza `K_MAX_CANDIDATOS = 101` como límite superior solicitado. Dentro de cada fold se aplica el filtrado de información mutua y correlación exclusivamente al subconjunto de entrenamiento del fold. A continuación, el notebook calcula el máximo número de descriptores que puede evaluarse en **todos los folds** y prueba `k = 1, 2, 3, ...` hasta ese máximo.

Esta limitación común evita que, cuando un fold tenga menos descriptores disponibles, varios valores altos de `k` terminen evaluando exactamente el mismo subconjunto truncado y distorsionen la curva. Por ello, el máximo efectivo puede ser menor que 101: depende de cuántos descriptores sobrevivan al filtrado en el fold con menor cantidad. El notebook imprime el rango efectivo antes de iniciar los ajustes.

La selección y el ranking se calculan dentro de los folds para evaluar la curva. Después, se genera un ranking final sobre el conjunto de entrenamiento completo para producir los artefactos que utilizan los notebooks posteriores. El conjunto de prueba reservado no se utiliza en el barrido de CV.

## Modelos y configuraciones

El flujo compara Ridge, Random Forest, XGBoost y LightGBM. Los notebooks posteriores generan las configuraciones finales a partir de la selección de descriptores y de la optimización documentada en el propio proyecto.

Los nombres «interpretable» y «potencia bruta» se usan como etiquetas internas de configuraciones; no implican por sí mismos interpretabilidad formal ni que se utilicen todos los descriptores disponibles.

## Instalación

Se recomienda instalar [Miniforge](https://github.com/conda-forge/miniforge) o una distribución compatible con Conda. Desde la raíz del repositorio:

```bash
conda env create -f environment.yml
conda activate sci_env
jupyter lab
```

También puede iniciarse el entorno clásico con `jupyter notebook`.

## Reproducción recomendada

1. Crear el entorno a partir de `environment.yml`.
2. Ejecutar los notebooks del 01 al 07 en orden.
3. Verificar que los archivos generados en `resultados/` corresponden a la ejecución actual.
4. Confirmar que las decisiones de selección y optimización no usan los conjuntos de evaluación externa.
5. Actualizar las tablas y métricas documentadas después de regenerar los resultados.

## Limitaciones

- SC-2 contiene pocas muestras, por lo que sus métricas pueden ser sensibles a la composición del conjunto.
- El rendimiento en un conjunto no garantiza generalización a otros dominios químicos.
- Para sostener conclusiones sobre bajo costo computacional, deben registrarse hardware, versiones, tiempos de ejecución y condiciones de comparación.
- La evaluación externa debe mantenerse separada de las decisiones de selección de variables, hiperparámetros y modelo.

## Referencias

1. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701

2. Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324

3. Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

5. Chen, Y., & Yang, Y. (2021). The one standard error rule for model selection: Does it work? *Stats, 4*(4), 868–892. https://doi.org/10.3390/stats4040051

6. Conn, J. G. M., Carter, J. W., Conn, J. J. A., Subramanian, V., Baxter, A., Engkvist, O., ... Palmer, D. S. (2023). Blinded predictions and post hoc analysis of the Second Solubility Challenge data: Exploring training data and feature set selection for machine and deep learning models. *Journal of Chemical Information and Modeling, 63*(4), 1099–1113. https://doi.org/10.1021/acs.jcim.2c01189

7. Delaney, J. S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. *Journal of Chemical Information and Computer Sciences, 44*(3), 1000–1005. https://doi.org/10.1021/ci034243x

8. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). New York, NY: Springer. https://doi.org/10.1007/978-0-387-84858-7

9. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634

10. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems, 30* (pp. 3146–3154). https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

11. Landrum, G. (s. f.). *RDKit: Open-source cheminformatics*. https://www.rdkit.org/

12. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html

13. Sorkun, M. C., Khetan, A., & Er, S. (2019). AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. *Scientific Data, 6*, Article 143. https://doi.org/10.1038/s41597-019-0151-1
