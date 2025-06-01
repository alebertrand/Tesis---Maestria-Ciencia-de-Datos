# Tesis-Maestria-Ciencia-de-Datos-Bertrand-Seijas

# Modelo Predictivo para Riesgo de Incumplimiento Crediticio con Metadata de Dispositivos Android

Este repositorio contiene el código utilizado para desarrollar un modelo predictivo orientado a anticipar el incumplimiento de pagos en créditos al consumo, usando exclusivamente metadata recolectada de dispositivos móviles Android al momento de la originación del crédito.

## Resumen del Proyecto

La motivación central de este proyecto es superar la limitación común en los modelos tradicionales de scoring crediticio, que dependen fuertemente de historiales financieros formales, excluyendo así a un amplio segmento de la población sin información en bureaus crediticios. Este trabajo explora la utilización de técnicas avanzadas de machine learning como regresión logística, Random Forest, XGBoost, LightGBM y redes neuronales, evaluando el valor predictivo de variables relacionadas con las características del dispositivo, configuración de seguridad y aplicaciones instaladas.

Se realizaron técnicas avanzadas de preprocesamiento, imputación y selección de variables, además de optimizar hiperparámetros con Optuna. Se abordó específicamente el desafío del desbalance del dataset utilizando múltiples estrategias de sobremuestreo mediante SMOTE.
