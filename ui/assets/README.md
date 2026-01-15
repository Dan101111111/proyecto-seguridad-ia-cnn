# Assets

Esta carpeta contiene recursos estáticos para la interfaz de usuario.

## Contenido:

- `styles.css` - Estilos personalizados para la aplicación Streamlit
- Imágenes de placeholder o logos (agregar según necesidad)
- Iconos personalizados
- Otros recursos visuales

## Uso:

Los estilos CSS se pueden cargar en Streamlit usando:

```python
with open('ui/assets/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
```
