"""
OBS: Esse formato deve ser adotado quando o usuário pretende treinar algum modelo e versioná-lo.
Para realizar inferência, e levar o modelo escolhido para produção, o formato será outro.
"""


import mlflow
from ultralytics import YOLO
import sys

class YOLO_OFERTA_SALE(mlflow.pyfunc.PythonModel):
    def __init__(self, path_model):
        # Carregar o modelo YOLO para detecção de objetos
        try:
            self.model_oferta = YOLO(path_model)
        except:
            print("\nERRO: Não foi possível encontrar os pesos para o modelo Ofertas em './Yolo Ofertas v1/best.pt'")
            sys.exit(1)

    def predict(self, img_path):
        results = self.model_oferta(img_path)
        return results
    

class YOLO_RECORTE_SALE(mlflow.pyfunc.PythonModel):
    def __init__(self, path_model):
        # Processar os dados e recortar as regiões de interesse aqui, conforme o código anterior
        try:
            self.model_recorte = YOLO(path_model)
        except:
            print("\nERRO: Não foi possível encontrar os pesos para o modelo Recortes em './Yolo Recortes v1/best.pt'")
            sys.exit(1)

    def predict(self, img_path):
        results = self.model_recorte(img_path)
        return results