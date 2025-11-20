"""
Archivo app.py para compatibilidad con Railway.
Railway busca automáticamente app.py o main.py si no encuentra un Procfile.
Este archivo importa y expone la aplicación Flask desde application.py
"""
from application import application

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)

