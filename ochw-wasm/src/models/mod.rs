use candle_core::{DType, Device, Error, Result, Tensor};

pub mod mobilenetv2;
pub mod sequential;

fn load_image64_raw(raw: Vec<u8>, device: &Device) -> Result<Tensor> {
    let data = Tensor::from_vec(raw, (64, 64, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.485f32, 0.456, 0.406], device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229f32, 0.224, 0.225], device)?.reshape((3, 1, 1))?;
    (data.to_dtype(DType::F32)? / 255.0)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

pub fn load_image_from_buffer(buffer: &[u8], device: &Device) -> Result<Tensor> {
    let img = image::load_from_memory(buffer)
        .map_err(Error::wrap)?
        .resize_to_fill(64, 64, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    
    load_image64_raw(img.into_raw(), device)
}

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn get_labels<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut labels = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if let Some(label) = line.split('\t').next() {
            labels.push(label.to_string());
        }
    }
    Ok(labels)
}


#[cfg(test)]
mod test {
    use std::{any, path::Path};

    use anyhow::Ok;
    use candle_nn::{ops::softmax, Module, VarBuilder};

    use super::*;

    #[test]
    fn it_works()->anyhow::Result<()> {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("ochw_mobilenetv2.safetensors");
        
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_path.as_path()],
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
            )
            .unwrap()
        };
        
        let nclasses = 4037;
        let model = mobilenetv2::Mobilenetv2::new(vb, nclasses).unwrap();

        let image_data = include_bytes!("../../../testdata/tu.png");
        let device = &Device::Cpu;
        let image = load_image_from_buffer(image_data, device).unwrap();
        let image = image.unsqueeze(0).unwrap();
        let output = model.forward(&image).unwrap();
        //softmax
        // top 5, candle 好像没有类似的 torch.topk 的函数，只能自己实现
        let output = softmax(&output, 1).unwrap();
        println!("{output}");
        // 获取 top 5 预测结果（包含索引和概率值）
        let mut predictions = output
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
            
        predictions.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let label_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/label.txt");
        let labels = get_labels(label_path).unwrap();
        
        let top5 = predictions.iter().take(5).collect::<Vec<_>>();
        for (i, (class_idx, prob)) in top5.iter().enumerate() {
            println!("{}. Class {}: {:.2}%", i+1, labels[*class_idx], prob * 100.0);
        }

        Ok(())
        
        
    }
}
