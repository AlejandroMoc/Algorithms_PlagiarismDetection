from PIL import Image
import torch # type: ignore
import torchvision.transforms as transforms # type: ignore
from torchvision import models # type: ignore

def preprocess_image(image_path):
    """Prepare and preprocess an image for model inference"""
    # Load image
    img = Image.open(image_path)
    
    # Define preprocessing steps
    preprocess_steps = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Preprocess image
    input_tensor = preprocess_steps(img)
    input_batch = input_tensor.unsqueeze(0)
    
    return input_batch

def make_prediction(my_model, input_tensor, class_labels):
    """Generate prediction on preprocessed image"""
    # Set model to evaluation mode
    my_model.eval()
    
    # Move input to device if GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_model = my_model.to(device)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = my_model(input_tensor)
        
    # Get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    results = []
    for i in range(5):
        results.append({
            'category': class_labels[top5_catid[i]],
            'probability': float(top5_prob[i])
        })
    
    return results

def execute():
    # Load pretrained model
    my_resnet_model = models.resnet50(pretrained=True)
    
    # Load ImageNet class names
    with open('imagenet_classes.txt', 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]
    
    # Example usage
    image_path = 'path/to/your/image.jpg'
    input_batch = preprocess_image(image_path)
    predictions = make_prediction(my_resnet_model, input_batch, class_labels)
    
    # Print results
    print("\nTop 5 predictions:")
    for pred in predictions:
        print(f"{pred['category']}: {pred['probability']:.4f}")

if __name__ == "__main__":
    execute()