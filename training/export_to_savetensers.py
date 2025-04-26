
from safetensors.torch import save_model
from trainer import HandwritingTrainer


if __name__ == "__main__":

    model_name = "mobilenetv2"

    model = HandwritingTrainer.load_from_checkpoint(
        f"./logs/{model_name}/version_2/checkpoint-epoch=12-val_loss=0.206.ckpt", model=model_name)
    model = model.model.base_model
    model.eval()
    print(model, file=open("model.txt", "w"))
    save_model(model, f"../ochw-wasm/ochw_{model_name}.safetensors")
    model.half()
    save_model(model, f"../ochw-wasm/ochw_{model_name}_fp16.safetensors")
