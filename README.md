# Proyecto TIC: Reconstrucción 3D Monocular con IA (2025)

Repositorio central para el proyecto de titulación sobre reconstrucción 3D monocular, enfocado en mejorar el sistema DeepDSO con módulos de IA.

## Estructura

- **/1_docs**: Documentación formal del proyecto (propuesta, informes, etc.).
- **/2_benchmarks**: Scripts y código para evaluar los modelos DCDepth y PixelFormer.
- **/3_deepdso_slam**: El sistema principal funcional, con el cliente C++ (DeepDSO) y el servidor Python (Monodepth2).
- **/4_android_app**: Futuro desarrollo de la aplicación para Android.
- **/external**: Scripts y enlaces para configurar dependencias externas pesadas (datasets).

## Guía de Configuración Inicial

1.  **Clonar el Repositorio (con LFS)**
    ```bash
    git clone [https://github.com/SebasPM15/TIC-2025.git](https://github.com/SebasPM15/TIC-2025.git)
    cd TIC-2025
    git lfs pull
    ```

2.  **Configurar el Dataset**
    El dataset EngelBenchmark no está en el repositorio. Ejecuta el siguiente script para ver instrucciones sobre cómo enlazar tu dataset local:
    ```bash
    ./external/downloader.sh
    ```

3.  **Configurar y Ejecutar los Componentes**
    Cada componente principal tiene su propio `README.md` con instrucciones detalladas de compilación y ejecución.