import streamlit as st
import streamlit.components.v1 as components
import os

st.title("Water Tank (Frontend) - Embedded")
st.markdown("This page embeds the static vanilla JS frontend for the Water Tank problem.")

# Serve local file — for local dev, you can embed file content:
# Relative path from backend/pages/ to frontend/water_tank/index.html
html_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "water_tank", "index.html")
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # We need to inline the CSS and JS for components.html to work seamlessly without serving static files via a separate server if dependencies are local
    # However, since the CSS and JS are separate files, we should read them and inject them into the HTML or use an iframe with a local server.
    # A simpler way for this single-file demo is to just read them and inject them into the HTML string before rendering.
    
    css_path = os.path.join(os.path.dirname(html_path), "styles.css")
    js_path = os.path.join(os.path.dirname(html_path), "app.js")
    
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css = f.read()
            html = html.replace('<link rel="stylesheet" href="styles.css" />', f'<style>{css}</style>')
            
    if os.path.exists(js_path):
        with open(js_path, "r") as f:
            js = f.read()
            # Replace the script tag src
            html = html.replace('<script src="app.js"></script>', f'<script>{js}</script>')

    components.html(html, height=700, scrolling=True)
else:
    st.error("Frontend HTML file not found.")
