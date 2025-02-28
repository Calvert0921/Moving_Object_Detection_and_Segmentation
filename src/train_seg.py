import os
import torch
import torch.optim as optim
from tqdm import tqdm

from model import MultiTaskModel
from dataloader import get_dataloader
from evaluate import iou_loss, generate_anchors
import torchvision.transforms as transforms

###########################################
# Training and Validation Functions       #
###########################################

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    # 定义交叉熵损失函数（适用于 segmentation 的 logits 与标签）
    ce_criterion = torch.nn.CrossEntropyLoss()
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", unit="batch")
    for i, batch in progress_bar:
        images = batch['images'].to(device)   # (B, 3, H, W)
        masks = batch['masks'].to(device)       # (B, H, W) 类别标签：0～5
        
        optimizer.zero_grad()
        seg_out = model(images)  # 模型输出 logits, shape: (B, 6, H, W)
        
        # 计算交叉熵损失
        ce_loss = ce_criterion(seg_out, masks)
        # 计算 IoU 损失
        iou_loss_val = iou_loss(seg_out, masks, num_classes=6)
        # 联合损失，权重可以根据需要调整；这里简单相加
        loss = ce_loss + 2*iou_loss_val
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        progress_bar.set_postfix({
            "CE Loss": f"{ce_loss.item():.4f}",
            "IoU Loss": f"{iou_loss_val.item():.4f}",
        })
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, device):
    model.eval()
    seg_iou_sum = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            seg_out = model(images)
            # 计算 IoU 损失，并转换为 IoU 指标（假设 IoU loss 取值在 [0,1] 内）
            seg_loss_val = iou_loss(seg_out, masks, num_classes=6)
            seg_iou = 1 - seg_loss_val.item()
            seg_iou_sum += seg_iou
            num_batches += 1
            progress_bar.set_postfix({"Seg IoU": f"{seg_iou:.4f}"})
    mean_seg_iou = seg_iou_sum / num_batches if num_batches > 0 else 0.0
    
    return mean_seg_iou

###########################################
# Main Training Routine                   #
###########################################

def main():
    # Hyperparameters.
    num_epochs = 50
    learning_rate = 1e-3
    batch_size = 4

    # Directories.
    train_images_dir = '../data/CamVid/train'
    train_masks_dir = '../data/annotations/train'
    train_annotations_file = '../data/annotations/train_anns.json'

    val_images_dir = '../data/CamVid/val'
    val_masks_dir = '../data/annotations/val'
    val_annotations_file = '../data/annotations/val_anns.json'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel()
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = get_dataloader(train_images_dir, train_masks_dir, train_annotations_file,
                                  batch_size=batch_size, num_workers=4, transform=transform, shuffle=True)
    val_loader = get_dataloader(val_images_dir, val_masks_dir, val_annotations_file,
                                batch_size=batch_size, num_workers=4, transform=transform, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 预生成 anchors（检测部分暂未使用）
    anchors = generate_anchors((45, 60), 16, [32, 64, 128]).to(device)

    best_acc = 0
    checkpoint_dir = "../weights"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"test_model.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"\nEpoch [{epoch}] Training Loss: {train_loss:.4f}")
        seg_iou = validate(model, val_loader, device)
        if seg_iou > best_acc:
            # 保存表现更好的模型
            torch.save(model.state_dict(), checkpoint_path)
            best_acc = seg_iou
            print(f"Saved checkpoint with better performance\n")
        print(f"Epoch [{epoch}] Validation Seg IoU: {seg_iou:.4f}")

if __name__ == '__main__':
    main()
