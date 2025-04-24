import { useEffect, useRef, useState } from "react";
import { Stage, Layer, Line, Rect } from "react-konva";
import Konva from "konva";

const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});

type ToolType = "brush" | "eraser";

interface LineData {
  tool: ToolType;
  points: number[];
}

interface CandidateWord{
  label:string;
  score: number;
  class_idx:number;
}

function App() {
  const poetry = `白日依山尽`;
  const poetryArray2 = ["", "", "", "", ""];
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [tool, _setTool] = useState<ToolType>("brush");
  const [lines, setLines] = useState<LineData[]>([]);
  const isDrawing = useRef(false);
  const [candidateWords, setCandidateWords] = useState<CandidateWord[]>([]);

  useEffect(() => {
    worker.onmessage = (e) => {
      // console.log(e.data);
      if (e.data && e.data.status =='complete') {
        setCandidateWords(e.data.output);
        // console.log(e.data.output)
      }
      // 
    };
  }, []);

  function cleanHandle(){
    setLines([])
    setCandidateWords([])
  }

  const handleMouseDown = (
    e: Konva.KonvaEventObject<MouseEvent> | Konva.KonvaEventObject<TouchEvent>
  ) => {
    isDrawing.current = true;

    const pos = e.target.getStage()?.getPointerPosition();
    if (pos) {
      setLines([...lines, { tool, points: [pos.x, pos.y] }]);
    }
  };

  const handleMouseMove = (
    e: Konva.KonvaEventObject<MouseEvent> | Konva.KonvaEventObject<TouchEvent>
  ) => {
    // no drawing - skipping
    if (!isDrawing.current) {
      return;
    }
    const stage = e.target.getStage();
    const point = stage?.getPointerPosition();

    if (!point) {
      return;
    }
    // To draw line
    const lastLine = lines[lines.length - 1];
    // add point
    lastLine.points = lastLine.points.concat([point.x, point.y]);

    // replace last
    lines.splice(lines.length - 1, 1, lastLine);
    setLines(lines.concat());
  };

  const handleMouseUp = async (
    e: Konva.KonvaEventObject<MouseEvent> | Konva.KonvaEventObject<TouchEvent>
  ) => {
    isDrawing.current = false;
    const stage = e.target.getStage();
    if (stage) {
      const blob: Blob = (await stage.toBlob()) as Blob;
      const arrayBuffer = await blob.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      // console.log(uint8Array)
      worker.postMessage({
        uint8Array,
        width: stage.width(),
        height: stage.height(),
      });
      //  debug:下载图片
      // const dataURL = stage.toDataURL({
      //   pixelRatio: 1, // double resolution
      // });

      // create link to download
      // const link = document.createElement("a");
      // link.download = "stage.png";
      // link.href = dataURL;
      // document.body.appendChild(link);
      // link.click();
      // document.body.removeChild(link);
    }
  };

  return (
    <div className="relative h-full">
      <div className="flex flex-col items-center justify-center w-full mx-auto py-8">
        <div className="grid grid-cols-5 border-1 border-indigo-500/20 rounded-xs">
          {poetry.split("").map((char, index) => (
            <div
              className="relative flex items-center justify-center border-1 border-indigo-500/20 divide-dashed divide-indigo-500/20 w-16 h-16"
              key={index}
            >
              <span className="inline-block text-2xl">{char}</span>
            </div>
          ))}
          {poetryArray2.map((char, index) => (
            <div
              className="relative flex items-center justify-center border-1 border-indigo-500/20 divide-dashed divide-indigo-500/20 w-16 h-16"
              key={index}
            >
              <span className="inline-block text-2xl">{char}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="absolute right-0 left-0 bottom-0 w-full h-1/2 bg-indigo-500/10  overflow-hidden">
        <div className="p-2 flex gap-1 items-center border-b-1 border-indigo-500/10">
          <div className="flex gap-1 flex-1">
            {candidateWords.map((word, index) => (
              <button
                key={index}
                type="button"
                className="bg-indigo-500/10 border-1 border-indigo-500/20 rounded-md px-2 py-1"
              >
                {word.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={cleanHandle}
            className="bg-rose-500/10 border-1 border-rose-500/20 rounded-md px-2 py-1 text-rose-700"
          >
            Clear
          </button>
        </div>
        <div>
          <Stage
            width={window.innerWidth}
            height={window.innerHeight / 2 - 60}
            onMouseDown={handleMouseDown}
            onMousemove={handleMouseMove}
            onMouseup={handleMouseUp}
            onTouchStart={handleMouseDown}
            onTouchMove={handleMouseMove}
            onTouchEnd={handleMouseUp}
          >
            <Layer>
              <Rect
                width={window.innerWidth}
                height={window.innerHeight / 2 - 60}
                fill="white"
              />
              {lines.map((line, i) => (
                <Line
                  key={i}
                  points={line.points}
                  stroke="#555555"
                  strokeWidth={6}
                  tension={0.5}
                  lineCap="round"
                  lineJoin="round"
                  globalCompositeOperation={
                    line.tool === "eraser" ? "destination-out" : "source-over"
                  }
                />
              ))}
            </Layer>
          </Stage>
        </div>
      </div>
    </div>
  );
}

export default App;
