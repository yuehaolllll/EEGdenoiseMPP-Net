import torch
import time
import os
import numpy as np
from tqdm import tqdm
from loss_function import denoise_loss_mse


def test_step(model, noiseEEG_test, EEG_test, device):
    """
    对应原代码的 test_step
    """
    model.eval()
    noiseEEG_test, EEG_test = noiseEEG_test.to(device), EEG_test.to(device)

    with torch.no_grad():
        denoiseoutput_test = model(noiseEEG_test)
        loss = denoise_loss_mse(denoiseoutput_test, EEG_test)

    return denoiseoutput_test, loss.item()


def train_step(model, noiseEEG_batch, EEG_batch, optimizer, device, criterion):
    model.train()
    noiseEEG_batch, EEG_batch = noiseEEG_batch.to(device), EEG_batch.to(device)
    optimizer.zero_grad()
    outputs = model(noiseEEG_batch)

    # 使用传入的 criterion，而不是硬编码的 denoise_loss_mse
    loss = criterion(outputs, EEG_batch)

    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, train_loader, val_x, val_y, epochs, optimizer, device, save_path, criterion):
    train_mse_history = []
    val_mse_history = []
    val_mse_min = 100.0
    os.makedirs(save_path, exist_ok=True)
    val_x, val_y = val_x.to(device), val_y.to(device)

    for epoch in range(epochs):
        model.train()
        train_mse = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for bx, by in pbar:
            # 传入 criterion
            batch_loss = train_step(model, bx, by, optimizer, device, criterion)
            train_mse += batch_loss

        model.eval()
        with torch.no_grad():
            v_out = model(val_x)
            # 验证集也使用相同的 criterion 或 统一使用 MSE (建议统一用 MSE 方便监控)
            avg_val_mse = torch.nn.functional.mse_loss(v_out, val_y).item()

        avg_train_mse = train_mse / len(train_loader)
        train_mse_history.append(avg_train_mse)
        val_mse_history.append(avg_val_mse)
        print(f" - Train Loss: {avg_train_mse:.6f}, Val MSE: {avg_val_mse:.6f}")

        if epoch > epochs * 0.8 and avg_val_mse < val_mse_min:
            val_mse_min = avg_val_mse
            torch.save(model.state_dict(), os.path.join(save_path, "denoise_model.pth"))
            print(">>> Best model saved!")

    history = {'loss': {'train_mse': train_mse_history, 'val_mse': val_mse_history}}
    return history