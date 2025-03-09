import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

#############################################
# Helper Functions for Detection Components #
#############################################

def generate_anchors(feature_size, stride, scales):
    """
    Generate anchors for a feature map.
    Args:
        feature_size (tuple): (height, width) of the feature map.
        stride (int): stride from input to feature map.
        scales (list of float): list of anchor scales.
    Returns:
        anchors (Tensor): shape (num_anchors, 4) in [xmin, ymin, xmax, ymax].
    """
    H, W = feature_size
    anchors = []
    for i in range(H):
        for j in range(W):
            center_x = (j + 0.5) * stride
            center_y = (i + 0.5) * stride
            for scale in scales:
                w = scale
                h = scale
                anchors.append([
                    center_x - w / 2, 
                    center_y - h / 2,
                    center_x + w / 2, 
                    center_y + h / 2
                ])
    return torch.tensor(anchors)  # shape (H*W*len(scales), 4)

def decode_boxes(anchors, bbox_deltas):
    """
    Decode predicted bbox deltas to boxes.
    Args:
        anchors (Tensor): shape (N, 4) in [xmin, ymin, xmax, ymax].
        bbox_deltas (Tensor): shape (N, 4) in [tx, ty, tw, th].
    Returns:
        pred_boxes (Tensor): shape (N, 4) in [xmin, ymin, xmax, ymax].
    """
    widths  = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = bbox_deltas[:, 0]
    dy = bbox_deltas[:, 1]
    dw = bbox_deltas[:, 2]
    dh = bbox_deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(bbox_deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    Args:
        boxes1 (Tensor): shape (N, 4)
        boxes2 (Tensor): shape (M, 4)
    Returns:
        iou (Tensor): shape (N, M)
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    boxes1 = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2 = boxes2.unsqueeze(0).expand(N, M, 4)

    inter_xmin = torch.max(boxes1[:, :, 0], boxes2[:, :, 0])
    inter_ymin = torch.max(boxes1[:, :, 1], boxes2[:, :, 1])
    inter_xmax = torch.min(boxes1[:, :, 2], boxes2[:, :, 2])
    inter_ymax = torch.min(boxes1[:, :, 3], boxes2[:, :, 3])

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
    area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
    union_area = area1 + area2 - inter_area + 1e-6
    iou = inter_area / union_area
    return iou

def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression.
    Args:
        boxes (Tensor): shape (N, 4)
        scores (Tensor): shape (N,)
        iou_threshold (float): threshold for suppression.
    Returns:
        keep (Tensor): indices of boxes to keep.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    
    # If boxes is 1D, unsqueeze to (1,4)
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
        scores = scores.unsqueeze(0)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    _, order = scores.sort(descending=True)
    # Ensure 'order' is 1D.
    if order.dim() == 0:
        order = order.unsqueeze(0)
    
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        order = order[1:]
        xx1 = torch.max(x1[i], x1[order])
        yy1 = torch.max(y1[i], y1[order])
        xx2 = torch.min(x2[i], x2[order])
        yy2 = torch.min(y2[i], y2[order])
        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[order] - inter + 1e-6)
        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze()
        # Ensure inds is 1D even when only one element is present.
        if inds.dim() == 0:
            inds = inds.unsqueeze(0)
        if inds.numel() == 0:
            break
        order = order[inds]
        if order.dim() == 0:
            order = order.unsqueeze(0)
    return torch.tensor(keep, dtype=torch.long)





##########################################
# Detection Loss and mAP Computation       #
##########################################

def compute_detection_loss(bbox_pred, cls_pred, gt_bboxes, gt_labels, anchors, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    """
    Compute detection loss using our custom implementation.
    Args:
        bbox_pred (Tensor): shape (B, num_anchors*4, H, W)
        cls_pred (Tensor): shape (B, num_anchors*num_classes, H, W)
        gt_bboxes (list of Tensors): each Tensor is (N, 4) in [xmin, ymin, w, h]
        gt_labels (list of Tensors): each Tensor is (N,) with class IDs (1-5; background=0)
        anchors (Tensor): shape (total_num_anchors, 4) in [xmin, ymin, xmax, ymax]
    Returns:
        loss (Tensor): detection loss.
    """
    device = bbox_pred.device
    num_classes = 6  # including background
    B = bbox_pred.shape[0]
    # Reshape predictions.
    bbox_pred = bbox_pred.view(B, -1, 4)         # (B, N, 4)
    cls_pred = cls_pred.view(B, -1, num_classes)   # (B, N, num_classes)
    
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_pos = 0
    cls_loss_fn = nn.CrossEntropyLoss(reduction='sum')
    reg_loss_fn = nn.SmoothL1Loss(reduction='sum')
    
    for i in range(B):
        # Get ground truth boxes and labels for image i.
        gt_boxes = gt_bboxes[i]  # (N,4) in [xmin, ymin, w, h]
        if gt_boxes.numel() == 0:
            gt_boxes = torch.empty((0,4), device=device)
            gt_lbls = torch.empty((0,), dtype=torch.long, device=device)
        else:
            gt_boxes = gt_boxes.clone().to(device)
            # Convert to [xmin, ymin, xmax, ymax]
            gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
            gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
            gt_lbls = gt_labels[i].to(device)
        
        # Decode predicted boxes.
        pred_deltas = bbox_pred[i]  # (N,4)
        pred_boxes = decode_boxes(anchors.to(device), pred_deltas)  # (N,4)
        
        # Compute IoU between each anchor and each gt box.
        if gt_boxes.size(0) > 0:
            ious = compute_iou(anchors.to(device), gt_boxes)  # (N, num_gt)
            max_iou, argmax_iou = ious.max(dim=1)  # (N,)
        else:
            max_iou = torch.zeros(anchors.size(0), device=device)
            argmax_iou = torch.zeros(anchors.size(0), dtype=torch.long, device=device)
        
        # Initialize target labels: -1 means ignore.
        target_labels = -1 * torch.ones(anchors.size(0), dtype=torch.long, device=device)
        # Negative anchors.
        target_labels[max_iou < neg_iou_thresh] = 0
        # Positive anchors.
        pos_inds = max_iou >= pos_iou_thresh
        if pos_inds.sum() > 0 and gt_boxes.size(0) > 0:
            target_labels[pos_inds] = gt_lbls[argmax_iou[pos_inds]]
        # Ensure each gt box has at least one positive anchor.
        if gt_boxes.size(0) > 0:
            for j in range(gt_boxes.size(0)):
                anchor_iou = ious[:, j]
                best_anchor = anchor_iou.argmax()
                target_labels[best_anchor] = gt_lbls[j]
                pos_inds[best_anchor] = True
        
        # Classification loss: use anchors with valid labels.
        valid_inds = target_labels != -1
        if valid_inds.sum() > 0:
            cls_loss = cls_loss_fn(cls_pred[i][valid_inds], target_labels[valid_inds])
        else:
            cls_loss = torch.tensor(0.0, device=device)
        
        # Regression loss: only for positive anchors.
        if pos_inds.sum() > 0:
            pos_target_boxes = gt_boxes[argmax_iou[pos_inds]]
            pos_pred_boxes = pred_boxes[pos_inds]
            reg_loss = reg_loss_fn(pos_pred_boxes, pos_target_boxes)
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        total_cls_loss += cls_loss
        total_reg_loss += reg_loss
        total_pos += pos_inds.sum().item()
    
    if total_pos > 0:
        loss = (total_cls_loss + total_reg_loss) / total_pos
    else:
        loss = total_cls_loss
    return loss

def compute_map(model, dataloader, device, anchors, iou_threshold=0.5, score_threshold=0.05):
    """
    Compute a simplified mAP for detection.
    Args:
        model: our multi-task model.
        dataloader: validation dataloader.
        device: computation device.
        anchors (Tensor): precomputed anchors.
    Returns:
        mAP (float): mean average precision over classes 1-5.
    """
    model.eval()
    all_detections = {}  # image_id -> {cls: list of (box, score)}
    all_annotations = {}  # image_id -> {cls: list of boxes}
    image_counter = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing mAP", unit="batch"):
            images = batch['images'].to(device)
            batch_size = images.shape[0]
            _, bbox_pred, cls_pred = model(images)
            B = bbox_pred.shape[0]
            bbox_pred = bbox_pred.view(B, -1, 4)      # (B, N, 4)
            cls_pred = cls_pred.view(B, -1, 6)         # (B, N, num_classes)
            for i in range(B):
                image_id = image_counter
                image_counter += 1
                scores = F.softmax(cls_pred[i], dim=-1)  # (N, num_classes)
                detections = {}
                for cls in range(1, 6):  # ignore background (0)
                    cls_scores = scores[:, cls]
                    inds = (cls_scores > score_threshold).nonzero().squeeze()
                    if inds.numel() == 0:
                        detections[cls] = []
                        continue
                    cls_boxes = decode_boxes(anchors.to(device), bbox_pred[i])[inds]
                    cls_scores = cls_scores[inds]
                    keep = nms(cls_boxes, cls_scores, iou_threshold=iou_threshold)
                    final_boxes = cls_boxes[keep].cpu()
                    final_scores = cls_scores[keep].cpu()
                    detections[cls] = list(zip(final_boxes.tolist(), final_scores.tolist()))
                all_detections[image_id] = detections

                # Process ground truth for image.
                gt_boxes = batch['bboxes'][i].to(device)  # (N_gt, 4) in [xmin, ymin, w, h]
                if gt_boxes.numel() > 0:
                    gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
                    gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
                gt_labels = batch['labels'][i].to(device)
                annotations = {}
                for cls in range(1, 6):
                    inds = (gt_labels == cls).nonzero().squeeze()
                    if inds.numel() == 0:
                        annotations[cls] = []
                    else:
                        cls_gt_boxes = gt_boxes[inds].cpu().tolist()
                        annotations[cls] = cls_gt_boxes
                all_annotations[image_id] = annotations

    # Compute AP per class.
    average_precisions = {}
    for cls in range(1, 6):
        cls_detections = []
        cls_annotations = {}
        for image_id in all_detections.keys():
            cls_detections.extend([(image_id, box, score) for box, score in all_detections[image_id].get(cls, [])])
            cls_annotations[image_id] = all_annotations[image_id].get(cls, [])
        if sum(len(v) for v in cls_annotations.values()) == 0:
            average_precisions[cls] = 0.0
            continue
        cls_detections = sorted(cls_detections, key=lambda x: x[2], reverse=True)
        TP = []
        FP = []
        total_gt = sum([len(v) for v in cls_annotations.values()])
        detected = {image_id: [False]*len(cls_annotations[image_id]) for image_id in cls_annotations}
        for image_id, box, score in cls_detections:
            box = torch.tensor(box)
            gt_boxes = torch.tensor(cls_annotations[image_id]) if len(cls_annotations[image_id]) > 0 else torch.empty((0, 4))
            if gt_boxes.numel() == 0:
                FP.append(1)
                TP.append(0)
                continue
            ious = compute_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, max_idx = ious.max(0)
            if max_iou >= iou_threshold and not detected[image_id][max_idx]:
                TP.append(1)
                FP.append(0)
                detected[image_id][max_idx] = True
            else:
                TP.append(0)
                FP.append(1)
        TP = torch.tensor(TP, dtype=torch.float)
        FP = torch.tensor(FP, dtype=torch.float)
        cum_TP = torch.cumsum(TP, dim=0)
        cum_FP = torch.cumsum(FP, dim=0)
        recalls = cum_TP / (total_gt + 1e-6)
        precisions = cum_TP / (cum_TP + cum_FP + 1e-6)
        ap = 0.0
        for t in torch.linspace(0, 1, steps=11):
            p = precisions[recalls >= t].max() if (recalls >= t).sum() > 0 else 0
            ap += p / 11
        average_precisions[cls] = ap
    mAP = sum(average_precisions.values()) / len(average_precisions)
    return mAP

#############################################
# IoU Loss for Segmentation                 #
#############################################

def iou_loss(pred, target, num_classes=6, smooth=1e-6):
    """
    Computes a soft IoU (Jaccard) loss for segmentation.
    Args:
        pred (Tensor): logits from segmentation head, shape (B, num_classes, H, W).
        target (Tensor): ground truth mask, shape (B, H, W).
    Returns:
        loss (Tensor): 1 - mean IoU.
    """
    pred_prob = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  # (B, H, W, num_classes)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()           # (B, num_classes, H, W)
    intersection = (pred_prob * target_one_hot).sum(dim=(2, 3))
    union = (pred_prob + target_one_hot - pred_prob * target_one_hot).sum(dim=(2, 3))
    iou = (intersection + smooth) / (union + smooth)
    loss = 1 - iou.mean()
    return loss