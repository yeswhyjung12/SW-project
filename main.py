import torch
import torch as nn
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# 이미지 파일 열기 및 전처리
imize = 300  # 이미지의 크기 설정
preprocess = transforms.Compose(
    [
        transforms.Resize(imize),
        transforms.CenterCrop(imize),  # 이미지의 가운데 부분을 자름
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# width = 320
# height = 320
# preprocess = transforms.Compose([
#     transforms.Resize((height, width)),  # 가로와 세로 크기를 다르게 설정
#     transforms.CenterCrop((height, width)),  # 가로와 세로 크기로 중심부를 자름
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# 이미지 torch.Tensor로 변환
def image_loader(image_path, device):  # Add device as a parameter
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    return image.to(device, torch.float)


image_path = "images/4.jpg"
use_cuda = True
image = image_loader(image_path, device)


# torch.Tensor이미지 출력
def imshow(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    # show an image whose values are between [0, 1]
    plt.imshow(image)


plt.figure()
plt.axis("off")  # 축 제거
imshow(image)

# 입력 데이터 정규화를 위한 클래스 정의
# !pip install torch torchvision torchaudio


# 입력 데이터 정규화를 위한 클래스 정의
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # Move mean and std to GPU during initialization
        self.mean = torch.Tensor(mean).to("cuda")
        self.std = torch.Tensor(std).to("cuda")

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


# 정규화 레이어를 포함한 ResNet50 모델 정의
model = (
    nn.Sequential(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
    )
    .to("cuda")
    .eval()
)  # GPU로 모델을 옮기기 및 평가(테스트) 모드로 변경

# ImageNet 클래스 레이블 불러오기
imagenet_labels = []  # ImageNet 클래스 레이블 리스트
with open("imagenet_labels.txt") as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

# 모델 예측 수행
image = image.to("cuda")  # Move image tensor to GPU
outputs = model(image)
percentages = (
    torch.nn.functional.softmax(outputs, dim=1)[0] * 100
)  # Use 'outputs' instead of 'output'
_, indices = torch.sort(outputs, descending=True)
top5_indices = indices[0][:5]
# 상위 5개 예측 결과 출력
print("가장 높은 확률을 가지는 클래스들:")
# Iterate directly over the elements of top5_indices
for i in top5_indices:
    print(
        f"인덱스: {i.item()} / 클래스명: {imagenet_labels[i.item()]} / 확률: {percentages[i].item():.2f}%"
    )

# Shadow Attack 함수 정의
def _tv_diff(channel):
  x_wise = channel[:,:,1:] - channel[:,:,:-1]
  y_wise = channel[:,1:,:] - channel[:,:-1,:]
  return (torch.sum(torch.abs(x_wise)) + torch.sum(torch.abs(y_wise))) # Return the sum directly

def smooth_ty(channel) -> torch.Tensor:
  tv_diff = _tv_diff(channel) # Get the total variation difference
  return tv_diff * tv_diff # Square the total variation difference

def get_tv(input: torch.Tensor) -> torch.Tensor:
  return smooth_ty(input[:,0,:,:]) + smooth_ty(input[:,1,:,:]) + smooth_ty(input[:,2,:,:]) # Assuming input is 4D (batch, channel, height, width)

def get_ct(input):
  return input.repeat((1, 3,  1, 1))

def shadow_attack_Linf(model, images, labels, targeted, eps, alpha, iters):
    # 이미지와 레이블을 GPU로 이동
    images = images.to(device)
    labels = labels.to(device)

    # 입력 이미지와 동일한 크기 및 채널을 가진 초기 노이즈 생성
    perturbation = torch.empty_like(images[:, :1, :, :]).uniform_(-eps, eps)
    perturbation = perturbation.to(device)

    # 손실 함수 설정 (교차 엔트로피)
    attack_loss = nn.CrossEntropyLoss()

    for i in range(iters):
        # requires_grad 속성 설정 (기울기 계산을 위해)
        perturbation.requires_grad = True

        # 그림자(Shadow)의 같은 형태를 위해 3채널로 복제하기
        ct = get_ct(perturbation)

        # 현재 공격 이미지(현재 이미지에 노이즈를 추가) 계산
        current = torch.clamp(images + ct, min=0, max=1)
        outputs = model(current)  # 모델의 예측 수행

        # 손실 함수 계산 및 기울기 계산
        model.zero_grad()
        cost = attack_loss(outputs, labels).to(device) + get_tv(ct) * 0.01  # TV(Total Variation) 손실 추가
        cost.backward()

        # 기울기를 사용하여 노이즈 업데이트
        if targeted:  # 타겟이 있는 공격일 경우
            diff = -alpha * perturbation.grad.sign()
        else:  # 타겟이 없는 공격일 경우
            diff = alpha * perturbation.grad.sign()

        # 노이즈 업데이트 및 클램핑
        perturbation.data = torch.clamp(perturbation + diff, min=-eps, max=eps).detach()
        perturbation.grad.zero_()  # 기울기 초기화

    # 만들어진 공격 이미지, 노이즈 반환
    current = torch.clamp(images + perturbation, min=0, max=1)
    return current, get_ct(perturbation)  # 수정된 부분

## 여기까지는 동일

# Target Attack
# 여기부터
# 공격용 파라미터 설정
targeted = True
eps = 0.01  # 입실론 값 설정 (노이즈 강도)
alpha = 1/255  # 노이즈 업데이트 스텝 사이즈
iters = 200  # 공격 반복 횟수

# 대상 클래스 레이블 설정
label = [18]  # 예: 18번 클래스
label = torch.Tensor(label).type(torch.long).to(device)  # 레이블을 텐서로 변환하고 GPU로 이동

# Shadow Attack 수행하여 공격 이미지 생성
adv_image, perturbation = shadow_attack_Linf(model, image, label, targeted, eps, alpha, iters)

# 공격된 이미지로 모델 예측 수행
outputs = model(adv_image)

# Softmax 함수를 통해 확률 계산
percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

# 상위 5개 예측 결과 출력
print("가장 높은 확률을 가지는 클래스들:")
for i in outputs[0].topk(5)[1]:
     print(f"인덱스: {i.item()} / 클래스명: {imagenet_labels[i.item()]} / 확률: {round(percentages[i].item(), 4)}%")

# 최종 공격 이미지 출력
plt.figure()
plt.axis('off')
imshow(adv_image)

# 노이즈 시각화
plt.figure()
plt.axis('off')
imshow(((perturbation + eps) / 2) / eps)

from PIL import Image

# 이미지 텐서를 PIL 이미지로 변환
result_image = transforms.ToPILImage()(adv_image.squeeze(0))

# 이미지 파일로 저장
result_image.save("output_image.png")
# 여기까지

# Unrestricted Attack
# 여기부터
# 공격용 파라미터 설정
targeted = False  # Unrestricted Attack이므로 Targeted를 False로 설정
eps = 0.03  # 입실론 값 설정 (노이즈 강도) - 필요에 따라 증가
alpha = 1/255  # 노이즈 업데이트 스텝 사이즈
iters = 200  # 공격 반복 횟수

# Assuming 'true_labels' is a tensor containing the correct labels for your images
true_labels = torch.tensor([18]).to(device) # Replace ... with the actual labels

# Shadow Attack 수행하여 공격 이미지 생성
adv_image, perturbation = shadow_attack_Linf(model, image, true_labels, targeted, eps, alpha, iters)

# 공격된 이미지로 모델 예측 수행
outputs = model(adv_image)

# Softmax 함수를 통해 확률 계산
percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

# 상위 5개 예측 결과 출력
print("가장 높은 확률을 가지는 클래스들:")
for i in outputs[0].topk(5)[1]:
    print(f"인덱스: {i.item()} / 클래스명: {imagenet_labels[i.item()]} / 확률: {round(percentages[i].item(), 4)}%")

# 최종 공격 이미지 출력
plt.figure()
plt.axis('off')
plt.imshow(adv_image.cpu().squeeze().permute(1, 2, 0).detach().numpy()) # 텐서를 NumPy 배열로 변환한 후 표시

# 노이즈 시각화
plt.figure()
plt.axis('off')
plt.imshow(((perturbation + eps) / (2 * eps)).cpu().squeeze().permute(1, 2, 0).detach().numpy()) # 텐서를 NumPy 배열로 변환한 후 표시
plt.show()

from PIL import Image

# 이미지 텐서를 PIL 이미지로 변환
result_image = transforms.ToPILImage()(adv_image.squeeze(0))

# 이미지 파일로 저장
result_image.save("output_image.png")
# 여기까지