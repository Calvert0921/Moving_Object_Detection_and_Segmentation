import os
import torch
import torch.optim as optim
from tqdm import tqdm

from model import MultiTaskModel
from dataloader import get_dataloader
from evaluate import iou_loss, compute_detection_loss, generate_anchors, compute_map
import torchvision.transforms as transforms

###########################################
# Training and Validation Functions       #
###########################################

def train_one_epoch(model, dataloader, optimizer, device, epoch, anchors, lambda_seg=1.0, lambda_det=1.0):
    model.train()
    running_loss = 0.0
    # class_weights = torch.tensor([1.0, 0.8, 1.2, 1.0, 1.0, 4]).to(device)
    # ce_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    ce_criterion = torch.nn.CrossEntropyLoss()

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", unit="batch")
    for i, batch in progress_bar:
        images = batch['images'].to(device)   # (B, 3, H, W)
        masks = batch['masks'].to(device)       # (B, H, W)
        gt_bboxes = batch['bboxes']             # list of Tensors (N,4) in [xmin, ymin, w, h]
        gt_labels = batch['labels']             # list of Tensors (N,)

        optimizer.zero_grad()
        # Forward pass: Even though all outputs are computed,
        # we'll only calculate the necessary loss below.
        seg_out, bbox_pred, cls_pred = model(images)
        
        # Check for the valid configuration
        if lambda_seg == 0.0 and lambda_det == 0.0:
            raise ValueError("Both lambda_seg and lambda_det cannot be zero.")

        # Compute only the required loss:
        if lambda_seg != 0.0 and lambda_det == 0.0:
            # Segmentation-only training.
            ce_loss = ce_criterion(seg_out, masks)
            seg_iou_loss = iou_loss(seg_out, masks, num_classes=6)
            seg_loss = ce_loss + 2 * seg_iou_loss
            loss = seg_loss
            pb_dict = {"Total Loss": f"{loss.item():.4f}", "Seg Loss": f"{seg_loss.item():.4f}"}
        elif lambda_seg == 0.0 and lambda_det != 0.0:
            # Detection-only training.
            det_loss = compute_detection_loss(bbox_pred, cls_pred, gt_bboxes, gt_labels, anchors)
            loss = det_loss
            pb_dict = {"Total Loss": f"{loss.item():.4f}", "Det Loss": f"{det_loss.item():.4f}"}
        elif lambda_seg != 0.0 and lambda_det != 0.0:
            # Joint training: compute both losses.
            ce_loss = ce_criterion(seg_out, masks)
            seg_iou_loss = iou_loss(seg_out, masks, num_classes=6)
            seg_loss = ce_loss + 2 * seg_iou_loss
            det_loss = compute_detection_loss(bbox_pred, cls_pred, gt_bboxes, gt_labels, anchors)
            loss = lambda_seg * seg_loss + lambda_det * det_loss
            pb_dict = {
                "Total Loss": f"{loss.item():.4f}",
                "Seg Loss": f"{seg_loss.item():.4f}",
                "Det Loss": f"{det_loss.item():.4f}"
            }
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(pb_dict)
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, device, anchors, phase):
    model.eval()
    seg_iou_sum = 0.0
    num_batches = 0
    det_loss_sum = 0.0
    progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            gt_bboxes = batch['bboxes'] 
            gt_labels = batch['labels']  
            
            seg_out, bbox_pred, cls_pred = model(images)
            
            # Phase 1: Segmentation
            if phase == 1:
                seg_loss_val = iou_loss(seg_out, masks, num_classes=6)
                seg_iou = 1 - seg_loss_val.item()
                seg_iou_sum += seg_iou
                num_batches += 1
                progress_bar.set_postfix({"Seg IoU": f"{seg_iou:.4f}"})
            # Phase 2: Detection
            elif phase == 2:
                detection_loss = compute_detection_loss(bbox_pred, cls_pred, gt_bboxes, gt_labels, anchors)
                det_loss_sum += detection_loss
                num_batches += 1
                progress_bar.set_postfix({"Det Loss": f"{detection_loss:.4f}"})
            # Phase 3: Combined training
            else:
                seg_loss_val = iou_loss(seg_out, masks, num_classes=6)
                seg_iou = 1 - seg_loss_val.item()
                seg_iou_sum += seg_iou
                progress_bar.set_postfix({"Seg IoU": f"{seg_iou:.4f}"})
                
                detection_loss = compute_detection_loss(bbox_pred, cls_pred, gt_bboxes, gt_labels, anchors)
                det_loss_sum += detection_loss
                num_batches += 1
                progress_bar.set_postfix({"Det Loss": f"{detection_loss:.4f}"})
    mean_seg_iou = seg_iou_sum / num_batches if num_batches > 0 else 0.0
    # detection_map = compute_map(model, dataloader, device, anchors)
    mean_det_loss = det_loss_sum / num_batches if num_batches > 0 else 0.0
    return mean_seg_iou, mean_det_loss

###########################################
# Main Training Routine                   #
###########################################

def main():
    # Hyperparameters.
    seg_epochs = 30         # Phase 1: segmentation training epochs
    det_epochs = 30         # Phase 2: detection training epochs
    joint_epochs = 30       # Phase 3: joint training epochs
    learning_rate = 1e-3
    batch_size = 4
    early_stop_patience = 5  # Epochs with no improvement before stopping early

    # Directories.
    train_images_dir = '../data/CamVid/train'
    train_masks_dir = '../data/annotations/train_masks'
    train_annotations_file = '../data/annotations/train_anns.json'

    val_images_dir = '../data/CamVid/val'
    val_masks_dir = '../data/annotations/val_masks'
    val_annotations_file = '../data/annotations/val_anns.json'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel()
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    
    train_loader = get_dataloader(train_images_dir, train_masks_dir, train_annotations_file,
                                  batch_size=batch_size, num_workers=4, transform=transform, shuffle=True)
    val_loader = get_dataloader(val_images_dir, val_masks_dir, val_annotations_file,
                                batch_size=batch_size, num_workers=4, transform=transform, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    anchors = generate_anchors((45, 60), 16, [32, 64, 128]).to(device)

    checkpoint_dir = "../weights"
    os.makedirs(checkpoint_dir, exist_ok=True)

    ############################################
    # Phase 1: Segmentation Only Training      #
    ############################################
    print("\n=== Phase 1: Segmentation Training ===")
    best_seg_iou = 0.0
    early_stop_counter_seg = 0
    seg_checkpoint_path = os.path.join(checkpoint_dir, "segmentation_model.pth")
    
    for epoch in range(1, seg_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, anchors, lambda_seg=1.0, lambda_det=0.0)
        print(f"\nPhase 1 - Epoch [{epoch}] Training Loss: {train_loss:.4f}")
        
        seg_iou, _ = validate(model, val_loader, device, anchors, phase=1)
        print(f"Phase 1 - Epoch [{epoch}] Validation Seg IoU: {seg_iou:.4f}")
        
        if seg_iou > best_seg_iou:
            torch.save(model.state_dict(), seg_checkpoint_path)
            best_seg_iou = seg_iou
            early_stop_counter_seg = 0
            print(f"Saved segmentation checkpoint at epoch {epoch} with IoU {seg_iou:.4f}\n")
        else:
            early_stop_counter_seg += 1
            print(f"No improvement in segmentation IoU for {early_stop_counter_seg} epoch(s).\n")
        
        if early_stop_counter_seg >= early_stop_patience:
            print(f"Early stopping segmentation training at epoch {epoch}. Best Seg IoU: {best_seg_iou:.4f}\n")
            break
    torch.cuda.empty_cache()

    ############################################
    # Phase 2: Detection Only Training         #
    ############################################
    print("\n=== Phase 2: Detection Training ===")
    best_det_loss = 999999
    early_stop_counter_det = 0
    det_checkpoint_path = os.path.join(checkpoint_dir, "detection_model.pth")
    
    for epoch in range(1, det_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, anchors, lambda_seg=0.0, lambda_det=1.0)
        print(f"\nPhase 2 - Epoch [{epoch}] Training Loss: {train_loss:.4f}")
        
        _, det_loss = validate(model, val_loader, device, anchors, phase=2)
        print(f"Phase 2 - Epoch [{epoch}] Validation Detection Loss: {det_loss:.4f}")
        
        if det_loss < best_det_loss:
            torch.save(model.state_dict(), det_checkpoint_path)
            best_det_loss = det_loss
            early_stop_counter_det = 0
            print(f"Saved detection checkpoint at epoch {epoch} with Loss {det_loss:.4f}\n")
        else:
            early_stop_counter_det += 1
            print(f"No improvement in detection Loss for {early_stop_counter_det} epoch(s).\n")
        
        if early_stop_counter_det >= early_stop_patience:
            print(f"Early stopping detection training at epoch {epoch}. Best Loss: {best_det_loss:.4f}\n")
            break
    torch.cuda.empty_cache()

    ############################################
    # Phase 3: Joint Training (Total Loss)     #
    ############################################
    print("\n=== Phase 3: Joint Training ===")
    joint_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    
    for epoch in range(1, joint_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, anchors, lambda_seg=40.0, lambda_det=1.0)
        print(f"\nPhase 3 - Epoch [{epoch}] Training Loss: {train_loss:.4f}")
        
        seg_iou, det_loss = validate(model, val_loader, device, anchors, phase=3)
        print(f"Phase 3 - Epoch [{epoch}] Validation Seg IoU: {seg_iou:.4f}  Detection Loss: {det_loss:.4f}")
        
        torch.save(model.state_dict(), joint_checkpoint_path)
        print(f"Saved joint training checkpoint at epoch {epoch}.\n")

if __name__ == '__main__':
    main()
