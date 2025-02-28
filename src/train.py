import os
import torch
import torch.optim as optim
from tqdm import tqdm

from model import MultiTaskModel
from dataloader import get_dataloader
from evaluate import iou_loss, compute_detection_loss, generate_anchors
import torchvision.transforms as transforms

###########################################
# Training and Validation Functions       #
###########################################

def train_one_epoch(model, dataloader, optimizer, device, epoch, anchors, lambda_seg=1.0, lambda_det=1.0):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", unit="batch")
    for i, batch in progress_bar:
        images = batch['images'].to(device)   # (B, 3, H, W)
        masks = batch['masks'].to(device)       # (B, H, W)
        gt_bboxes = batch['bboxes']              # list of Tensors (N,4) in [xmin, ymin, w, h]
        gt_labels = batch['labels']              # list of Tensors (N,)
        
        optimizer.zero_grad()
        seg_out, bbox_pred, cls_pred = model(images)
        seg_loss = iou_loss(seg_out, masks, num_classes=6)
        det_loss = compute_detection_loss(bbox_pred, cls_pred, gt_bboxes, gt_labels, anchors)
        loss = lambda_seg * seg_loss + lambda_det * det_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({
            "Total Loss": f"{loss.item():.4f}",
            "Seg IoU Loss": f"{seg_loss.item():.4f}",
            "Det Loss": f"{det_loss.item():.4f}"
        })
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, device, anchors):
    model.eval()
    seg_iou_sum = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            seg_out, bbox_pred, cls_pred = model(images)
            seg_loss_val = iou_loss(seg_out, masks, num_classes=6)
            seg_iou = 1 - seg_loss_val.item()
            seg_iou_sum += seg_iou
            num_batches += 1
            progress_bar.set_postfix({"Seg IoU": f"{seg_iou:.4f}"})
    mean_seg_iou = seg_iou_sum / num_batches if num_batches > 0 else 0.0
    # detection_map = compute_map(model, dataloader, device, anchors)
    detection_map = 0
    return mean_seg_iou, detection_map

###########################################
# Main Training Routine                   #
###########################################

def main():
    # Hyperparameters.
    num_epochs = 50
    learning_rate = 1e-3
    batch_size = 4

    # Directories.
    train_images_dir = 'data/CamVid/train'
    train_masks_dir = 'data/annotations/train'
    train_annotations_file = 'data/annotations/trian_anns.json'

    val_images_dir = 'data/CamVid/val'
    val_masks_dir = 'data/annotations/val'
    val_annotations_file = 'data/annotations/val_anns.json'

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

    # Precompute anchors based on feature map size (45x60), stride 16, and chosen scales.
    anchors = generate_anchors((45, 60), 16, [32, 64, 128]).to(device)

    best_acc = 0
    checkpoint_dir = "weights"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, anchors)
        print(f"\nEpoch [{epoch}] Training Loss: {train_loss:.4f}")
        seg_iou, det_map = validate(model, val_loader, device, anchors)
        if seg_iou > best_acc:
            # Save best model
            torch.save(model.state_dict(), checkpoint_path)
            best_acc = seg_iou
            print(f"Saved checkpoint with better performance\n")
        print(f"Epoch [{epoch}] Validation Seg IoU: {seg_iou:.4f}  Detection mAP: {det_map:.4f}")

if __name__ == '__main__':
    main()