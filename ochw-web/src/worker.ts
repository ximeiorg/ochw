import wasmUrl from "../node_modules/ochw-wasm/ochw_wasm_bg.wasm?url"
import init, { Model } from "ochw-wasm"
// 实现一个 HandingWrite 类的单例
export class HandingWrite {
    private static instance: Model | undefined;
    private constructor() { }
    public static async getInstance(): Promise<Model> {
        if (!this.instance) {
            self.postMessage({ status: `loading` });
            await init(wasmUrl)
            this.instance = Model.new()
            self.postMessage({ status: `loaded` });
            return this.instance;
        } else {
            return this.instance;
        }
    }
}


self.addEventListener("message", async (event: MessageEvent) => {
    try {
        const model = await HandingWrite.getInstance();
        const {uint8Array} = event.data;
        if (uint8Array.length === 0){
            self.postMessage({ status: `loaded` });
            return;
        }
        const res = model.predict(uint8Array);
        self.postMessage({
            status: "complete",
            output: JSON.parse(res),
        });
    }
    catch (e) {
        self.postMessage({ error: `worker error: ${e}` });
    }
})