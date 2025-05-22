python
from PIL import Image
import torch # type: ignore
import torchvision.transforms as transforms # type: ignore
from torchvision import models # type: ignore

def cargar_imagen(ruta_imagen):
    """Cargar y preprocesar una imagen para inferencia del modelo"""
    # Cargar imagen
    imagen = Image.open(ruta_imagen)
    
    # Definir pasos de preprocesamiento
    preproceso = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Preprocesar imagen
    tensor_entrada = preproceso(imagen)
    lote_entrada = tensor_entrada.unsqueeze(0)
    
    return lote_entrada

def predecir_imagen(modelo, tensor_imagen, nombres_clases):
    """Realizar predicción en imagen preprocesada"""
    # Establecer modelo en modo de evaluación
    modelo.eval()
    
    # Mover entrada a dispositivo si GPU disponible
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    tensor_imagen = tensor_imagen.to(dispositivo)
    
    with torch.no_grad():
        salida = modelo(tensor_imagen)
        
    # Obtener probabilidades
    probabilidades = torch.nn.functional.softmax(salida[0], dim=0)
    
    # Obtener las 5 principales predicciones
    top5_prob, top5_catid = torch.topk(probabilidades, 5)
    
    resultados = []
    for i in range(5):
        resultados.append({
            'categoria': nombres_clases[top5_catid[i]],
            'probabilidad': float(top5_prob[i])
        })
    
    return resultados

def principal():
    # Cargar modelo preentrenado
    modelo = models.resnet50(pretrained=True)
    
    # Cargar nombres de clases de ImageNet
    with open('imagenet_classes.txt', 'r') as f:
        nombres_clases = [line.strip() for line in f.readlines()]
    
    # Uso de ejemplo
    ruta_imagen = 'ruta/a/tu/imagen.jpg'
    lote_entrada = cargar_imagen(ruta_imagen)
    predicciones = predecir_imagen(modelo, lote_entrada, nombres_clases)
    
    # Imprimir resultados
    print("\nTop 5 predicciones:")
    for pred in predicciones:
        print(f"{pred['categoria']}: {pred['probabilidad']:.4f}")

if __name__ == "__main__":
    principal()