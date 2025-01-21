import numpy as np
import torch
import torchvision
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def saliency(image):
    input2 = torch.nn.functional.pad(image.clone().detach(), (3,3,3,3), mode='replicate')
    out = image.clone().detach()
    out = torch.abs(saliency.kernel(input2.to(device)))
    out = saliency.blur(out)
    out = torch.abs(out)
    att, _ = torch.max(out, dim=1)
    out[0, 0] = out[0, 1] = out[0, 2] = att
    out+=0.05
    out[out>1] = 1
    return out
saliency.blur = torch.nn.Conv2d(3, 3, 5, groups=3, padding='valid').to(device)
saliency.blur.weight = torch.nn.Parameter(torch.ones_like(saliency.blur.weight).to(device),requires_grad=False)
saliency.blur.weight[:,:,:,:] = 1./25.
saliency.kernel = torch.nn.Conv2d(3, 3, 3, groups=3, padding='valid').to(device)
saliency.kernel.weight = torch.nn.Parameter(torch.ones_like(saliency.kernel.weight).to(device),requires_grad=False)
saliency.kernel.weight[:,:,:,:] = -1
saliency.kernel.weight[:,:,1,1] = 8. 
print(f'{saliency.kernel.weight=}')
print(f'{saliency.blur.weight=}')

if __name__ == "__main__":
    for i in range(100):
        # Read image and make it a tensor
        image = cv2.imread("test.png")
        # image[:,:,:] = 127
        image[10:20,10:20,:] = 0
        image[10:20,50:60,:] = 255
        cv2.imwrite(f"test_input.png", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)/255

        # Process
        outimg = saliency(image.unsqueeze(dim=0)).squeeze()

        # Save
        outimg = np.transpose(outimg.cpu().detach().numpy(), [1,2,0])*255
        outimg = outimg.astype(np.uint8)
        outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"test_output_{str(i).zfill(6)}.png", outimg)    # Process
