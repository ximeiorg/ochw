
from safetensors import save_model
from training.trainer import HandwritingTrainer


if __name__ == "__main__":

    model_name = "mobilenetv2"

    model = HandwritingTrainer.load_from_checkpoint(
        f"./logs/{model_name}/version_0/checkpoint-epoch=32-val_loss=0.156.ckpt", model=model_name)
    model = model.model
    model.eval()
    print(model, file=open("model.txt", "w"))
    save_model(model, f"ochw_{model_name}.safetensors")
