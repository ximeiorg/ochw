use candle_core::{DType, Device, Result};
use candle_nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};

use crate::{models::{load_image_from_buffer, mobilenetv2::Mobilenetv2}, utils::auto_crop_image_content};

#[derive(Serialize, Deserialize)]
pub struct Topk {
    pub label: String,
    pub score: f32,
    pub class_idx:usize,
}

pub struct Inference {
    model: Mobilenetv2,
}

impl Inference {

    /// 加载并初始化一个预训练的 MobileNetV2 模型。
    pub fn load_model(weights: &[u8]) -> Result<Self> {
        let dev = &Device::Cpu;
        // let weights = include_bytes!("../ochw_mobilenetv2_fp16.safetensors");
        let vb = VarBuilder::from_buffered_safetensors(weights.to_vec(), DType::F32, dev)?;
        let model = Mobilenetv2::new(vb, 4037)?;
        Ok(Self { model })
    }

        /// 从指定的标签文件中获取所有标签，并将其作为字符串向量返回。
        /// ## 返回值
        /// - `Result<Vec<String>>`: 如果成功读取并解析文件，返回包含所有标签的 `Vec<String>`；
        ///   如果过程中发生错误（如文件读取失败或解析错误），返回相应的错误信息。
        pub fn get_labels(&self) -> Result<Vec<String>> {
            // 读取标签文件内容
            let label_text = include_str!("../label.txt");
            let reader = BufReader::new(label_text.as_bytes());
    
            let mut labels = Vec::new();
            // 逐行读取文件内容
            for line in reader.lines() {
                let line = line?;
                let line = line.trim();
                // 提取每行的第一个字段作为标签
                if let Some(label) = line.split('\t').next() {
                    labels.push(label.to_string());
                }
            }
            Ok(labels)
        }

    /// 使用预训练的模型对输入的图像进行预测，并返回概率最高的前5个类别及其对应的概率。
    ///
    /// # 参数
    /// - `image`: 输入的图像数据，以 `Vec<u8>` 形式表示，通常为图像的二进制数据。
    /// - `topk`: 选取前k个概率最高的类别，默认为5。
    ///
    /// # 返回值
    /// - `Result<Vec<Top5>>`: 返回一个包含前5个预测结果的 `Vec<Top5>`，每个 `Top5` 结构体包含类别标签、概率值和类别索引。
    ///   如果过程中出现错误，则返回 `Err`。
    pub fn predict(&self, image: Vec<u8>,topk:Option<usize>) -> anyhow::Result<Vec<Topk>> {
        // 从缓冲区加载图像并将其转换为模型所需的张量格式
        let image = auto_crop_image_content(&image)?;
        let image = load_image_from_buffer(&image, &Device::Cpu)?;
        let image = image.unsqueeze(0)?;
    
        // 使用模型对图像进行前向传播，获取输出结果
        let output = self.model.forward(&image)?;
    
        // 对输出结果进行 softmax 处理，将其转换为概率分布
        let output = candle_nn::ops::softmax(&output, 1)?;
    
        // 将输出结果展平并转换为包含索引和概率值的向量
        let mut predictions = output
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
    
        // 根据概率值对预测结果进行排序，从高到低
        predictions.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    
        // 获取类别标签
        let labels = self.get_labels()?;
        
        let topk = topk.unwrap_or(5);
        // 取概率最高的前5个预测结果
        let topk_data = predictions.iter().take(topk).collect::<Vec<_>>();
    
        // 将前5个预测结果转换为 `Top5` 结构体，并打印结果
        let mut top5_data = Vec::with_capacity(topk);
        for (i, (class_idx, prob)) in topk_data.iter().enumerate() {
            println!(
                "{}. Class {}: {:.2}%",
                i + 1,
                labels[*class_idx],
                prob * 100.0
            );
            top5_data.push(Topk {
                label: labels[*class_idx].clone(),
                score: *prob,
                class_idx: *class_idx,
            })
        }
    
        Ok(top5_data)
    }
}
