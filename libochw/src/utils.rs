use std::io::Cursor;

use image::{GenericImageView, ImageBuffer, Rgba};

/// 计算图像的 Otsu 阈值
pub fn otsu_threshold(hist: &[u32; 256]) -> u8 {
    let total: u64 = hist.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return 0;
    }

    let mut sum_back = 0_u64;
    let mut weight_back = 0_u64;
    let mut max_variance = 0_f32;
    let mut threshold = 0_u8;

    let intensities: [f32; 256] = std::array::from_fn(|i| i as f32);

    let mut sum_fore = hist
        .iter()
        .zip(intensities.iter())
        .map(|(&v, &i)| v as f32 * i)
        .sum::<f32>();
    let mut weight_fore = total as f32;

    for i in 0..256 {
        weight_back += hist[i] as u64;
        if weight_back == 0 {
            continue;
        }
        weight_fore -= hist[i] as f32;
        if weight_fore == 0.0 {
            break;
        }

        sum_back += hist[i] as u64 * i as u64;
        sum_fore -= hist[i] as f32 * intensities[i];

        let mean_back = sum_back as f32 / weight_back as f32;
        let mean_fore = sum_fore / weight_fore;

        let variance = weight_back as f32 * weight_fore * (mean_back - mean_fore).powi(2);

        if variance > max_variance {
            max_variance = variance;
            threshold = i as u8;
        }
    }

    threshold
}

/// 图像边缘检测
pub fn auto_crop_image_content(buffer: &[u8]) -> anyhow::Result<Vec<u8>> {
    let img = image::load_from_memory(buffer)?;
    let (width, height) = img.dimensions();

    // 转换为灰度图
    let gray_img = img.grayscale();

    let hist = imageproc::stats::histogram(&gray_img.to_rgb8());

    let channel_hist = &hist.channels[0]; // 假设至少有一个通道（灰度图）
    let threshold = otsu_threshold(channel_hist);

    println!("Otsu's threshold: {}", threshold);

    let bin_img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = gray_img.get_pixel(x, y);
        if pixel[0] < threshold {
            Rgba([0, 0, 0, 255]) // 前景标记为黑色
        } else {
            Rgba([255, 255, 255, 255]) // 背景标记为白色
        }
    });

    // 查找内容边界
    let mut min_x = width;
    let mut max_x = 0;
    let mut min_y = height;
    let mut max_y = 0;

    for y in 0..height {
        for x in 0..width {
            let pixel = bin_img.get_pixel(x, y);
            if pixel[0] != 255 {
                // 检测二值化后的前景
                min_x = std::cmp::min(min_x, x);
                max_x = std::cmp::max(max_x, x);
                min_y = std::cmp::min(min_y, y);
                max_y = std::cmp::max(max_y, y);
            }
        }
    }

    // 边界检查
    if min_x >= width || min_y >= height {
        return Err(anyhow::anyhow!("未检测到有效内容"));
    }

    //优化框坐标，主要是为了让框更接近正方形，提升识别效果
    // 获取裁剪的高和宽
    let crop_width = max_x - min_x;
    let crop_height = max_y - min_y;

    //获取高宽差
    let diff = crop_height.abs_diff(crop_width);
    if diff > 0 {
        // 如果高度大于宽度，向左边和右边分别添加边距，但是不能超过图片边界
        if crop_height > crop_width {
            let new_min_x = std::cmp::max(min_x.saturating_sub(diff / 2), 0);
            let new_max_x = std::cmp::min(max_x.saturating_add(diff / 2), width);
            min_x = new_min_x;
            max_x = new_max_x;
        } else {
            let new_min_y = std::cmp::max(min_y.saturating_sub(diff / 2), 0);
            let new_max_y = std::cmp::min(max_y.saturating_add(diff / 2), height);
            min_y = new_min_y;
            max_y = new_max_y;
        }
    }

    // 执行裁剪并保存结果
    let mut cur = Cursor::new(vec![]);
    let crop_image = image::imageops::crop(
        &mut img.to_rgba8(),
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y,
    )
    .to_image();
    // .save_with_format(output_path, image::ImageFormat::Png)?;
    crop_image.write_to(&mut cur, image::ImageFormat::Png)?;

    Ok(cur.into_inner())
}

#[cfg(test)]
mod test {
    use std::fs;

    use super::*;

    #[test]
    fn test_find_contours() -> anyhow::Result<()> {
        let buffers = (
            include_bytes!("../test_data/han.png"),
            include_bytes!("../test_data/yi.png"),
            include_bytes!("../test_data/y1.png"),
        );

        let data = auto_crop_image_content(buffers.2)?;
        fs::write("test.png", data)?;

        Ok(())
    }
}
