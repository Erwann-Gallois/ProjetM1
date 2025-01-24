import subprocess
import time
import requests
import sys

def start_server():
    """Lance le serveur FastAPI via uvicorn dans le dossier fastAPI."""
    print("Lancement du serveur FastAPI...")
    try:
        process = subprocess.Popen(
            ["uvicorn", "main:app", "--reload"],
            cwd="fastAPI",  # Change le dossier courant pour exécuter depuis fastAPI
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Pour décoder les logs en texte directement
        )
        return process
    except Exception as e:
        print(f"Erreur lors du lancement du serveur : {e}")
        return None

def test_server():
    """Vérifie si le serveur FastAPI est prêt."""
    print("Vérification que le serveur est actif...")
    url = "http://127.0.0.1:8000"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Le serveur est actif et prêt.")
            return True
    except requests.exceptions.ConnectionError:
        print("Le serveur n'est pas encore prêt...")
    return False

def start_client():
    """Lance le script request.py."""
    print("Lancement du client (request.py)...")
    try:
        subprocess.run([sys.executable, "request.py"], cwd="fastAPI")
    except Exception as e:
        print(f"Erreur lors de l'exécution du client : {e}")

def main():
    server_process = start_server()
    if not server_process:
        print("Le serveur n'a pas pu être lancé.")
        return

    try:
        # Vérifie que le serveur est actif
        for _ in range(10):
            if test_server():
                break
            time.sleep(2)
        else:
            print("Le serveur n'est pas disponible après plusieurs essais.")
            return

        # Lancer le client une fois le serveur prêt
        start_client()
    finally:
        # Arrête le serveur proprement
        print("Arrêt du serveur...")
        server_process.terminate()

if __name__ == "__main__":
    main()
