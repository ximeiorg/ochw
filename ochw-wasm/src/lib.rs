mod utils;

use libochw::worker::Worker;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}
#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::worker::log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub fn run() {
    alert("Hello, ochw-wasm!");
}

#[wasm_bindgen]
pub struct Model {
    worker: Worker,
    labels: Vec<String>,
}

#[wasm_bindgen]
impl Model {
    pub fn new() -> Result<Self, JsError> {
        let weights = include_bytes!("../ochw_mobilenetv2_fp16.safetensors");
        let worker = Worker::load_model(weights)?;
        let labels = worker.get_labels()?;
        Ok(Self { worker, labels })
    }

    /// 获取标签
    pub fn get_label(&self) -> Result<String, JsError> {
        let json = serde_json::to_string(&self.labels)?;
        Ok(json)
    }

    /// 推理
    pub fn predict(&self, image: Vec<u8>) -> Result<String, JsError> {
        let output = self.worker.predict(image)?;
        let json = serde_json::to_string(&output)?;
        Ok(json)
    }
}
