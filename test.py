import torch
from RWTN_Net import RWTN_Net
from utils import Interpolation_Coefficient
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
weight_root = './weights'
device = torch.device('cuda:0')  # device 'cuda' or 'cpu'
Coefficient_3 = Interpolation_Coefficient(3)
Coefficient_3=Coefficient_3.to(device)
model = RWTN_Net(Coefficient_3=Coefficient_3).to(device)

weight_path = weight_root + r'\best.pth'
checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint['model'])
model.eval()

resize=320
tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize((int(resize), int(resize))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
path='./test.png'
with torch.no_grad():
    with torch.cuda.amp.autocast():
        img=tf(path).unsqueeze(0)
        img=img.to(device)
        result1,result2 = model(img)
    result = torch.sigmoid(result1)
    result = result[0][0].cpu().numpy()
    result = (result > 0.5).astype(int)

    filepath = "./test_result.png"

    plt.imsave(filepath, result, cmap='gray')