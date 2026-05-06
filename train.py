import torch
from tqdm import tqdm

from src.config import *
from src.model import GlassModel, EMA, freeze_bn, cosine_lr, set_seed
from src.dataset import GlassDataset, load_image_paths, get_sample_weights
from src.metrics import evaluate
from torch.utils.data import DataLoader, WeightedRandomSampler


def main():
    set_seed(SEED)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print("Current device:", device)

    train_paths = load_image_paths(TRAIN_DATASET_PATH)
    test_paths = load_image_paths(TEST_DATASET_PATH)
    train_paths = train_paths[:TRAIN_SIZE]
    test_paths = test_paths[:TEST_SIZE]
    print(f"Train set: {len(train_paths)} | Test set: {len(test_paths)}")

    train_dataset = GlassDataset(train_paths, train=True)
    test_dataset = GlassDataset(test_paths, train=False)

    sample_weights = get_sample_weights(train_paths)
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=use_amp,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=use_amp
    )

    model = GlassModel().to(device)
    ema = EMA(model)
    criterion = torch.nn.HuberLoss(delta=5.0 / MAX_SCORE)

    for param in model.backbone.features.parameters():
        param.requires_grad = False
    is_backbone_unfrozen = False

    backbone_params = list(model.backbone.features.parameters())
    head_params = list(model.backbone.classifier.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LEARNING_RATE_BACKBONE, "base_lr": LEARNING_RATE_BACKBONE},
            {"params": head_params, "lr": LEARNING_RATE, "base_lr": LEARNING_RATE}
        ],
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_test_mae = float('inf')
    best_test_acc5 = 0.0
    early_stop_counter = 0
    phase = 1

    model_save_path = CHECKPOINTS_DIR / MODEL_SAVE_NAME
    ema_model_save_path = CHECKPOINTS_DIR / EMA_MODEL_SAVE_NAME

    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS and phase == 1:
            print("\nUnfreezing backbone for full training")
            phase = 2
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": LEARNING_RATE_BACKBONE, "base_lr": LEARNING_RATE_BACKBONE},
                    {"params": head_params, "lr": LEARNING_RATE, "base_lr": LEARNING_RATE}
                ],
                weight_decay=WEIGHT_DECAY
            )
            early_stop_counter = 0
            print("Early stop counter reset")

        cosine_lr(optimizer, epoch, WARMUP_EPOCHS, EPOCHS)
        model.train()
        freeze_bn(model)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update()

            total_loss += loss.item() * ACCUM_STEPS
            n_steps += 1

        train_loss = total_loss / n_steps
        current_lr = optimizer.param_groups[0]['lr']

        test_metrics = evaluate(model, test_loader, device)

        print(f"\nEpoch {epoch + 1:2d}/{EPOCHS} | LR: {current_lr:.2e} | Train Loss: {train_loss:.3f}")
        print(
            f"Test set | MAE: {test_metrics['mae']:.2f} | RMSE: {test_metrics['rmse']:.2f} | Acc within 5: {test_metrics['acc_5pt']:.2%}")

        current_acc5 = test_metrics["acc_5pt"]
        current_mae = test_metrics["mae"]

        if (current_acc5 > best_test_acc5) or (current_acc5 == best_test_acc5 and current_mae < best_test_mae):
            best_test_acc5 = current_acc5
            best_test_mae = current_mae
            torch.save(model.state_dict(), model_save_path)
            ema.apply_shadow()
            torch.save(model.state_dict(), ema_model_save_path)
            ema.restore()
            print(f"Best model saved | Best test acc within 5: {best_test_acc5:.2%} | MAE: {best_test_mae:.2f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stop count: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print("\nEarly stop triggered, training terminated early")
                break

    print("\n" + "=" * 70)
    print("Training complete! Final results")
    print("=" * 70)
    print(f"Main model path: {model_save_path}")
    print(f"EMA smoothed model path: {ema_model_save_path}")
    print(f"Best test MAE: {best_test_mae:.2f}")
    print(f"Best test acc within 5: {best_test_acc5:.2%}")

    print("\nPer-grade performance of best model")
    print("=" * 70)
    model.load_state_dict(torch.load(model_save_path))
    final_metrics = evaluate(model, test_loader, device)

    score_bins = [55, 60, 65, 70, 75, 80, 85, 90]
    print(f"{'Grade':<6} | {'n (test)':<8} | {'MAE':<8} | {'acc±5':<6}")
    print("-" * 40)

    preds = final_metrics["preds"]
    targets = final_metrics["targets"]
    for bin_score in score_bins:
        mask = (targets == bin_score)
        if np.sum(mask) == 0:
            continue
        bin_mae = np.mean(np.abs(preds[mask] - targets[mask]))
        bin_acc5 = np.mean(np.abs(preds[mask] - targets[mask]) <= 5.0) * 100
        print(f"{bin_score:<6} | {np.sum(mask):<8} | {bin_mae:<8.2f} | {bin_acc5:.0f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()