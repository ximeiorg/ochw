use candle_core::{DType, Device, Result};
use candle_nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};

use crate::models::load_image_from_buffer;

#[derive(Serialize, Deserialize)]
pub struct Top5 {
    pub label: String,
    pub score: f32,
    pub class_idx:usize,
}

pub struct Worker {
    model: crate::models::mobilenetv2::Mobilenetv2,
}

impl Worker {
    pub fn load_model() -> Result<Self> {
        let dev = &Device::Cpu;
        let weights = include_bytes!("../ochw_mobilenetv2.safetensors");
        let vb = VarBuilder::from_buffered_safetensors(weights.to_vec(), DType::F32, dev)?;
        let model = crate::models::mobilenetv2::Mobilenetv2::new(vb, 4037)?;
        Ok(Self { model })
    }

    pub fn get_labels(&self) -> Result<Vec<String>> {
        let label_text = include_str!("../../training/data/train/label.txt");
        let reader = BufReader::new(label_text.as_bytes());

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

    pub fn predict(&self, image: Vec<u8>) -> Result<Vec<Top5>> {
        let image = load_image_from_buffer(&image, &Device::Cpu)?;
        let image = image.unsqueeze(0)?;
        let output = self.model.forward(&image)?;
        // top 5, candle 好像没有类似的 torch.topk 的函数，只能自己实现
        let output = candle_nn::ops::softmax(&output, 1)?;
        // 获取 top 5 预测结果（包含索引和概率值）
        let mut predictions = output
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();

        predictions.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let labels = self.get_labels()?;

        let top5 = predictions.iter().take(5).collect::<Vec<_>>();

        let mut top5_data = Vec::with_capacity(5);
        for (i, (class_idx, prob)) in top5.iter().enumerate() {
            println!(
                "{}. Class {}: {:.2}%",
                i + 1,
                labels[*class_idx],
                prob * 100.0
            );
            top5_data.push(Top5 {
                label: labels[*class_idx].clone(),
                score: *prob,
                class_idx: *class_idx,
            })
        }

        Ok(top5_data)
    }
}
