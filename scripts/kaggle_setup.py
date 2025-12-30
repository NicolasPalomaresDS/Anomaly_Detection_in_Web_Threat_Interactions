import os
import stat
from ipywidgets import FileUpload
from IPython.display import display
from pathlib import Path

def setup_kaggle_credentials():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    # Si ya está configurado, no hace falta subir nada
    if os.path.exists(kaggle_json_path):
        print("✅ kaggle.json already configured")
        return

    # Mostrar selector de archivos
    print("📁 upload your kaggle.json")
    uploader = FileUpload(accept='.json', multiple=False)
    display(uploader)

    def save_uploaded(uploader):
        if not uploader.value:
            print("⚠️ no file uploaded")
            return
        
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Guardar el archivo subido
        for file_info in uploader.value:
            with open(kaggle_json_path, 'wb') as f:
                f.write(file_info['content'])
            print("✅ kaggle.json uploaded correctly")
        
        # Cambiar permisos (solo lectura para el usuario actual)
        os.chmod(kaggle_json_path, stat.S_IRUSR | stat.S_IWUSR)
        print("🔒 permissions configured correctly")
        print("✅ kaggle API ready to use")

    # Esperar a que el usuario suba el archivo y luego guardarlo
    uploader.observe(lambda change: save_uploaded(uploader), names='value')
    

def download_data(link, path=Path('dataset')):
    if os.path.exists(path):
        print("✅ dataset already downloaded")
    else:
        os.makedirs("dataset", exist_ok=True)
        os.system(link)